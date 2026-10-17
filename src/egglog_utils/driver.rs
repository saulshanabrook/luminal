use super::*;
use crate::typed_reference;
use colored::Colorize;
use egglog_experimental::typed::report::{ReportDuration, ReportStageKind};
use itertools::Itertools;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use tracing::trace;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RewritePhase {
    Expr,
    IntervalExpr,
    DtypeProp,
    MatmulFlatten,
    KernelLower,
    DirectKernel,
    KernelSpecialize,
    BufferReuse,
    MatmulBackend,
    Glumoe,
    FusionPair,
    FusionGrow,
    FusionMerge,
    KernelFuseLatePre,
    KernelFuseLate,
    KernelFuseLate2,
    Cleanup,
    PostCleanup,
    BaseCleanup,
    LateAnalysis,
}

/// Rule groups at the scheduling boundaries of Luminal's optimization recipe.
/// These are ordinary immutable Egglog rulesets, not operation registrations.
pub type PhaseRules = BTreeMap<RewritePhase, Ruleset>;

pub fn base_phase_rules() -> PhaseRules {
    use RewritePhase::*;
    [
        (Expr, typed_reference::expressions.clone()),
        (
            IntervalExpr,
            typed_reference::static_rules::intervals.clone(),
        ),
        (DtypeProp, typed_reference::dynamic_rules::dtypes.clone()),
        (MatmulFlatten, typed_reference::static_rules::matmul.clone()),
        (KernelLower, ruleset(())),
        (DirectKernel, ruleset(())),
        (KernelSpecialize, ruleset(())),
        (BufferReuse, ruleset(())),
        (MatmulBackend, ruleset(())),
        (Glumoe, ruleset(())),
        (FusionPair, ruleset(())),
        (FusionGrow, ruleset(())),
        (FusionMerge, ruleset(())),
        (KernelFuseLatePre, ruleset(())),
        (KernelFuseLate, ruleset(())),
        (KernelFuseLate2, ruleset(())),
        (Cleanup, ruleset(())),
        (PostCleanup, ruleset(())),
        (BaseCleanup, typed_reference::static_rules::cleanup.clone()),
        (LateAnalysis, ruleset(())),
    ]
    .into_iter()
    .collect()
}

/// The production fixed point, also available to bounded schedule experiments.
pub fn main_schedule(phases: &PhaseRules, intervals: bool) -> Schedule {
    use RewritePhase::*;
    let expression = if intervals {
        sequence((&phases[&Expr], &phases[&IntervalExpr])).saturate()
    } else {
        phases[&Expr].saturate()
    };
    let producer = sequence((
        expression.clone(),
        phases[&DtypeProp].saturate(),
        &phases[&MatmulFlatten],
        &phases[&KernelLower],
        &phases[&DirectKernel],
        &phases[&KernelSpecialize],
        &phases[&BufferReuse],
        &phases[&MatmulBackend],
        &phases[&Glumoe],
        &phases[&FusionPair],
    ))
    .saturate();
    let fusion = sequence((
        expression,
        phases[&DtypeProp].saturate(),
        &phases[&FusionGrow],
        &phases[&FusionMerge],
    ))
    .saturate();
    sequence((producer, fusion)).saturate()
}

/// Final fusion and cleanup. Experiments may omit a phase without inspecting
/// rule bodies or duplicating the production recipe.
pub fn final_phases(phases: &PhaseRules, intervals: bool) -> [(RewritePhase, Schedule); 5] {
    use RewritePhase::*;
    let expression = if intervals {
        sequence((&phases[&Expr], &phases[&IntervalExpr])).saturate()
    } else {
        phases[&Expr].saturate()
    };
    // Relation cascade: invf(pre) → angles → rotation → concat. Three
    // semi-naive late runs are intentional, unlike saturation of this stage.
    let late_fusion = sequence((
        &phases[&KernelFuseLatePre],
        &phases[&KernelFuseLate],
        &phases[&KernelFuseLate],
        &phases[&KernelFuseLate],
        &phases[&KernelFuseLate2],
    ));
    [
        (KernelFuseLate, late_fusion),
        (Expr, expression),
        (Cleanup, phases[&Cleanup].saturate()),
        (PostCleanup, phases[&PostCleanup].saturate()),
        (BaseCleanup, phases[&BaseCleanup].saturate()),
    ]
}

const MAIN_SCHEDULE_MAX_CYCLES: usize = 256;
const SLOW_PHASE_TIME: Duration = Duration::from_secs(1);
const BIG_TUPLE_DELTA: isize = 5_000;

#[derive(Debug, Clone)]
struct EgglogSchedulePhase {
    name: String,
    schedule: String,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogStageReport {
    pub num_matches_per_rule: FxHashMap<String, usize>,
    pub search_and_apply_time_per_rule: FxHashMap<String, Duration>,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogRunReport {
    pub full: EgglogStageReport,
    pub phases: Vec<EgglogPhaseReport>,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogPhaseReport {
    pub name: String,
    pub schedule: String,
    pub updated: bool,
    pub iterations: usize,
    pub tuples_before: usize,
    pub tuples_after: usize,
    pub total_time: Duration,
}

// Display-only schedule descriptions preserve diagnostics; execution uses the
// corresponding native Schedule values, never parses these strings.
fn expr_schedule(use_interval_analysis: bool) -> &'static str {
    if use_interval_analysis {
        "(saturate (seq expr interval_expr))"
    } else {
        "(saturate expr)"
    }
}

fn egglog_main_schedule(use_interval_analysis: bool) -> String {
    let expr = expr_schedule(use_interval_analysis);
    // Producer rules create raw alternatives that downstream fusion consumes.
    // Fusion grow/merge only consumes Kernel*/FusionEnd alternatives, so keeping
    // producer discovery saturated before fusion reaches the same fixed point
    // while avoiding repeated expensive pair-discovery scans during growth.
    format!(
        "(saturate (seq
        (saturate (seq
            {expr}
            (saturate dtype_prop)
            (run matmul_flatten)
            (run kernel_lower)
            (run direct_kernel)
            (run kernel_specialize)
            (run buffer_reuse)
            (run matmul_backend)
            (run glumoe)
            (run fusion_pair)
        ))
        (saturate (seq
            {expr}
            (saturate dtype_prop)
            (run fusion_grow)
            (run fusion_merge)
        ))
    ))"
    )
}

/// Preserve the original report-driven outer cycle and final/late phase policy.
pub fn run_schedule(
    egraph: &mut EGraph,
    full_start: Instant,
    phases: &PhaseRules,
    intervals: bool,
    log: bool,
    setup: impl FnOnce(&mut EGraph) -> Result<(), TypedError>,
) -> Result<EgglogRunReport, TypedError> {
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED_DEBUG_EGGLOG: AtomicBool = AtomicBool::new(false);
        if log && !WARNED_DEBUG_EGGLOG.swap(true, Ordering::Relaxed) {
            eprintln!("{}", "   Egglog warning: running in a debug build; model-sized saturation can be very slow. Use `cargo run --release ...` for CUDA model examples.".yellow());
        }
    }
    let setup_tuples_before = egraph.num_tuples()?;
    let install_start = Instant::now();
    // A zero-repeat native schedule installs all referenced declarations/rules
    // without firing them. Keep installation out of the first timed phase.
    egraph.run(sequence(phases.values().collect::<Vec<_>>()).repeat(0))?;
    let install_elapsed = install_start.elapsed();
    let register_start = Instant::now();
    setup(egraph)?;
    let register_elapsed = register_start.elapsed();
    let setup_tuples_after = egraph.num_tuples()?;
    if log {
        eprintln!(
            "{}",
            format!(
                "   Egglog {:<28} {:>10} | install {} register {} | tuples {} -> {} ({:+})",
                "setup",
                metric_duration(full_start.elapsed()),
                metric_duration(install_elapsed),
                metric_duration(register_elapsed),
                setup_tuples_before,
                setup_tuples_after,
                setup_tuples_after as isize - setup_tuples_before as isize,
            )
            .cyan()
        );
    }
    let mut reports = Vec::new();
    let mut reached_fixed_point = false;
    for cycle in 1..=MAIN_SCHEDULE_MAX_CYCLES {
        let phase = EgglogSchedulePhase {
            name: format!("cycle {cycle:03} main"),
            schedule: egglog_main_schedule(intervals),
        };
        let updated = run_schedule_phase(
            egraph,
            &mut reports,
            &phase,
            main_schedule(phases, intervals),
            log,
        )?;
        if egraph.num_tuples()? > MAIN_SCHEDULE_MAX_TUPLES {
            return Err(TypedError::Invalid(format!(
                "egglog saturation exceeded tuple budget: {} > {MAIN_SCHEDULE_MAX_TUPLES}",
                egraph.num_tuples()?
            )));
        }
        if !updated {
            reached_fixed_point = true;
            break;
        }
    }
    if !reached_fixed_point {
        return Err(TypedError::Invalid(format!(
            "egglog saturation did not reach a fixed point within {MAIN_SCHEDULE_MAX_CYCLES} cycles"
        )));
    }
    for (phase, schedule) in final_phases(phases, intervals) {
        let (name, description) = match phase {
            RewritePhase::KernelFuseLate => (
                "fuse late",
                "(seq kernel_fuse_late_pre kernel_fuse_late kernel_fuse_late kernel_fuse_late kernel_fuse_late2)",
            ),
            RewritePhase::Expr => ("final expr", expr_schedule(intervals)),
            RewritePhase::Cleanup => ("cleanup", "(saturate cleanup)"),
            RewritePhase::PostCleanup => ("post cleanup", "(saturate post_cleanup)"),
            RewritePhase::BaseCleanup => ("base cleanup", "(saturate base_cleanup)"),
            _ => unreachable!(),
        };
        run_schedule_phase(
            egraph,
            &mut reports,
            &EgglogSchedulePhase {
                name: name.into(),
                schedule: description.into(),
            },
            schedule,
            log,
        )?;
    }
    if !phases[&RewritePhase::LateAnalysis].is_empty() {
        run_schedule_phase(
            egraph,
            &mut reports,
            &EgglogSchedulePhase {
                name: "late pass 01".into(),
                schedule: "(saturate metal_memory_analysis)".into(),
            },
            phases[&RewritePhase::LateAnalysis].saturate(),
            log,
        )?;
    }
    let full = stage_report(egraph, full_start.elapsed())?;
    trace_stage_report("---- Egglog Rule Matches ----", &full);
    let report = EgglogRunReport {
        full,
        phases: reports,
        total_time: full_start.elapsed(),
    };
    print_run_summary_with_log(&report, log);
    trace!(
        "{}",
        format!(
            "---- Egglog Total Took {} ----",
            metric_duration(report.total_time)
        )
        .green()
    );
    Ok(report)
}

fn stage_report(
    egraph: &mut EGraph,
    total_time: Duration,
) -> Result<EgglogStageReport, TypedError> {
    let run_report = egraph.stats()?;
    Ok(EgglogStageReport {
        num_matches_per_rule: run_report
            .detail
            .num_matches_per_rule
            .iter()
            .map(|entry| {
                (
                    entry.native_name.clone(),
                    usize::try_from(entry.value).expect("native match count fits usize"),
                )
            })
            .collect(),
        search_and_apply_time_per_rule: run_report
            .detail
            .search_and_apply_time_per_rule
            .iter()
            .map(|entry| {
                (
                    entry.native_name.clone(),
                    report_duration(entry.value.as_ref()),
                )
            })
            .collect(),
        total_time,
    })
}

fn trace_stage_report(header: &str, report: &EgglogStageReport) {
    trace!("{}", header.green());
    trace!(
        "{}",
        report
            .num_matches_per_rule
            .iter()
            .filter(|(k, _)| !k.contains("("))
            .map(|(k, v)| format!(
                "{k}: {v} ({})",
                pretty_duration::pretty_duration(&report.search_and_apply_time_per_rule[k], None)
            ))
            .join("\n")
            .green()
    );
    trace!(
        "{}",
        format!(
            "---- {} Took {} ----",
            header,
            pretty_duration::pretty_duration(&report.total_time, None).bold()
        )
        .green()
    );
}

fn metric_duration(duration: Duration) -> String {
    pretty_duration::pretty_duration(&duration, None)
}

fn report_duration(duration: Option<&ReportDuration>) -> Duration {
    let duration = duration.expect("typed reports contain required metric durations");
    Duration::new(duration.seconds, duration.nanos)
}

/// Extra per-rule / per-ruleset printing. Requires the normal egglog log
/// channel plus `EGGLOG_DEBUG=1`.
fn egglog_debug() -> bool {
    std::env::var("EGGLOG_DEBUG").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn metric_name(name: &str) -> String {
    let mut name = name.split_whitespace().join(" ");
    if name.len() > 96 {
        name.truncate(93);
        name.push_str("...");
    }
    name
}

fn sorted_rule_metrics(
    report: &egglog_experimental::typed::RunReport,
) -> Vec<(String, Duration, usize)> {
    let matches: FxHashMap<_, _> = report
        .detail
        .num_matches_per_rule
        .iter()
        .map(|entry| (entry.native_name.as_str(), entry.value))
        .collect();
    let mut rules = report
        .detail
        .search_and_apply_time_per_rule
        .iter()
        .map(|entry| {
            (
                entry.native_name.clone(),
                report_duration(entry.value.as_ref()),
                usize::try_from(
                    matches
                        .get(entry.native_name.as_str())
                        .copied()
                        .unwrap_or_default(),
                )
                .expect("native match count fits usize"),
            )
        })
        .collect_vec();
    rules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    rules
}

fn print_rule_plan_hotspots(
    report: &egglog_experimental::typed::RunReport,
    rules: &[(String, Duration, usize)],
) {
    for (rule, _, _) in rules.iter().take(3) {
        let mut max_stage = None;
        let mut max_shape = None;
        for iteration in &report.detail.iterations {
            let ruleset = iteration
                .rule_set_report
                .as_ref()
                .expect("typed iteration reports contain their ruleset report");
            for rule_reports in ruleset
                .rule_reports
                .iter()
                .filter(|entry| entry.native_name == *rule)
            {
                for rule_report in &rule_reports.reports {
                    if let Some(plan) = &rule_report.plan {
                        let scans = plan
                            .stages
                            .iter()
                            .map(|stage| {
                                match stage
                                    .kind
                                    .as_ref()
                                    .expect("typed plan stages contain their kind")
                                {
                                    ReportStageKind::Intersect(stage) => stage.scans.len(),
                                    ReportStageKind::FusedIntersect(stage) => {
                                        stage.to_intersect.len() + 1
                                    }
                                }
                            })
                            .sum::<usize>();
                        max_shape = Some(
                            max_shape
                                .map(|(stages, shape_scans)| {
                                    if plan.stages.len() > stages {
                                        (plan.stages.len(), scans)
                                    } else {
                                        (stages, shape_scans)
                                    }
                                })
                                .unwrap_or((plan.stages.len(), scans)),
                        );
                        for stage in &plan.stages {
                            if let Some(stats) = &stage.stats {
                                if max_stage
                                    .map(|(candidates, _)| stats.num_candidates > candidates)
                                    .unwrap_or(true)
                                {
                                    max_stage = Some((stats.num_candidates, stats.num_succeeded));
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some((stages, scans)) = max_shape {
            eprintln!(
                "      plan    {:<96} stages {:>2} scans {:>2} | max candidates {}",
                metric_name(rule),
                stages,
                scans,
                max_stage
                    .map(|(candidates, succeeded)| format!("{candidates} -> {succeeded}"))
                    .unwrap_or_else(|| "n/a".to_string())
            );
        }
    }
}

fn print_slow_phase_detail(
    phase: &EgglogSchedulePhase,
    report: &egglog_experimental::typed::RunReport,
    tuple_delta: isize,
    elapsed: Duration,
    rules: &[(String, Duration, usize)],
) {
    eprintln!("      detail  schedule {}", metric_name(&phase.schedule));
    if tuple_delta > 0 && elapsed > Duration::ZERO {
        eprintln!(
            "      detail  growth {:.0} tuples/s | {:.3} ms/new tuple",
            tuple_delta as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0 / tuple_delta as f64
        );
    }
    for (rule, elapsed, _) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches == 0)
        .take(5)
    {
        eprintln!(
            "      zero    {:<96} {:>10}",
            metric_name(rule),
            metric_duration(*elapsed)
        );
    }
    let mut per_match = rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches > 0)
        .map(|(rule, elapsed, matches)| {
            (
                rule,
                elapsed.as_secs_f64() * 1_000.0 / *matches as f64,
                *matches,
            )
        })
        .collect_vec();
    per_match.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rule, ms_per_match, matches) in per_match.into_iter().take(3) {
        eprintln!(
            "      cost    {:<96} {:.3} ms/match | matches {}",
            metric_name(rule),
            ms_per_match,
            matches
        );
    }
    print_rule_plan_hotspots(report, rules);

    let iteration_count = report.detail.iterations.len();
    for (index, iteration) in report.detail.iterations.iter().enumerate() {
        if iteration_count > 12 && (8..iteration_count.saturating_sub(3)).contains(&index) {
            if index == 8 {
                eprintln!("      iter   ...");
            }
            continue;
        }
        let ruleset = iteration
            .rule_set_report
            .as_ref()
            .expect("typed iteration reports contain their ruleset report");
        let (rule, elapsed, matches) = ruleset
            .rule_reports
            .iter()
            .map(|entry| {
                (
                    entry.native_name.clone(),
                    entry
                        .reports
                        .iter()
                        .map(|report| report_duration(report.search_and_apply_time.as_ref()))
                        .sum::<Duration>(),
                    entry
                        .reports
                        .iter()
                        .map(|report| {
                            usize::try_from(report.num_matches)
                                .expect("native match count fits usize")
                        })
                        .sum::<usize>(),
                )
            })
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)))
            .unwrap_or_else(|| ("-".to_string(), Duration::ZERO, 0));
        eprintln!(
            "      iter   {:>2} changed={} | search {:>10} | merge {:>10} | rebuild {:>10} | top {:<64} {:>10} matches {}",
            index + 1,
            ruleset.changed,
            metric_duration(report_duration(ruleset.search_and_apply_time.as_ref())),
            metric_duration(report_duration(ruleset.merge_time.as_ref())),
            metric_duration(report_duration(iteration.rebuild_time.as_ref())),
            metric_name(&rule),
            metric_duration(elapsed),
            matches
        );
    }
}

fn print_run_summary_with_log(run_report: &EgglogRunReport, log: bool) {
    if !log {
        return;
    }
    eprintln!(
        "{}",
        format!(
            "   Egglog summary total {} | phases {}",
            metric_duration(run_report.total_time),
            run_report.phases.len()
        )
        .cyan()
    );
    if !egglog_debug() {
        return;
    }
    let mut phases = run_report.phases.iter().collect_vec();
    phases.sort_by_key(|phase| std::cmp::Reverse(phase.total_time));
    for phase in phases.into_iter().take(5) {
        eprintln!(
            "      phase   {:<28} {:>10} | tuples {:+} | iterations {}",
            metric_name(&phase.name),
            metric_duration(phase.total_time),
            phase.tuples_after as isize - phase.tuples_before as isize,
            phase.iterations
        );
    }
    let mut growth = run_report
        .phases
        .iter()
        .map(|phase| {
            (
                phase,
                phase.tuples_after as isize - phase.tuples_before as isize,
            )
        })
        .filter(|(_, delta)| *delta > 0)
        .collect_vec();
    growth.sort_by_key(|(_, delta)| std::cmp::Reverse(*delta));
    for (phase, delta) in growth.into_iter().take(3) {
        eprintln!(
            "      growth  {:<28} tuples {:+} | {}",
            metric_name(&phase.name),
            delta,
            metric_duration(phase.total_time)
        );
    }
    let mut rules = run_report
        .full
        .search_and_apply_time_per_rule
        .iter()
        .map(|(rule, elapsed)| {
            (
                rule,
                *elapsed,
                run_report
                    .full
                    .num_matches_per_rule
                    .get(rule)
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect_vec();
    rules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    for (rule, elapsed, matches) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO || *matches > 0)
        .take(8)
    {
        eprintln!(
            "      slow    {:<96} {:>10} | matches {}",
            metric_name(rule),
            metric_duration(*elapsed),
            matches
        );
    }
    for (rule, elapsed, _) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches == 0)
        .take(8)
    {
        eprintln!(
            "      zero    {:<96} {:>10}",
            metric_name(rule),
            metric_duration(*elapsed)
        );
    }
}

/// Pre-cleanup diagnostic aggregate over native equality rows only. Scalars and
/// the removed OutputJoin are not fabricated to mimic serializer statistics.
pub fn trace_frozen_shape(space: &SearchSpace, log: bool) {
    if !log || !egglog_debug() {
        return;
    }
    let mut classes = 0;
    let mut nodes = 0;
    let mut labels: FxHashMap<String, usize> = FxHashMap::default();
    for class in space.frozen.eclasses() {
        let mut active = false;
        for node in class.nodes().filter(|node| !node.is_subsumed()) {
            active = true;
            nodes += 1;
            *labels.entry(node.name().to_owned()).or_default() += 1;
        }
        classes += usize::from(active);
    }
    let mut labels = labels.into_iter().collect_vec();
    labels.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    eprintln!("{}", format!(
        "   Egglog extract root shape equality_rows={} equality_classes={} output_roots={} top_constructors={}",
        nodes, classes, space.roots.len(),
        labels.into_iter().take(8).map(|(label, count)| format!("{}={count}", metric_name(&label))).join(", "),
    ).cyan());
}

fn run_schedule_phase(
    egraph: &mut EGraph,
    phases: &mut Vec<EgglogPhaseReport>,
    phase: &EgglogSchedulePhase,
    schedule: Schedule,
    log: bool,
) -> Result<bool, TypedError> {
    let tuples_before = egraph.num_tuples()?;
    let start = Instant::now();
    let report = egraph.run(schedule)?;
    let elapsed = start.elapsed();
    let tuples_after = egraph.num_tuples()?;

    let updated = report.detail.updated;
    let iterations = report.detail.iterations.len();
    let tuple_delta = tuples_after as isize - tuples_before as isize;
    if log {
        eprintln!(
            "{}",
            format!(
                "   Egglog {:<28} {:>10} | tuples {} -> {} ({:+}) | updated={} | iterations={}",
                phase.name,
                metric_duration(elapsed),
                tuples_before,
                tuples_after,
                tuple_delta,
                updated,
                iterations,
            )
            .cyan()
        );
    }

    if log && egglog_debug() {
        let mut rulesets: FxHashMap<_, [Duration; 3]> = FxHashMap::default();
        for (index, entries) in [
            &report.detail.search_and_apply_time_per_ruleset,
            &report.detail.merge_time_per_ruleset,
            &report.detail.rebuild_time_per_ruleset,
        ]
        .into_iter()
        .enumerate()
        {
            for entry in entries {
                rulesets.entry(entry.native_name.as_str()).or_default()[index] =
                    report_duration(entry.value.as_ref());
            }
        }
        let mut rulesets = rulesets.into_iter().collect_vec();
        rulesets.sort_by_key(|(_, durations)| {
            std::cmp::Reverse(durations.iter().copied().sum::<Duration>())
        });
        for (ruleset, [search, merge, rebuild]) in rulesets.into_iter().take(4) {
            eprintln!(
                "      ruleset {:<18} search {:>10} | merge {:>10} | rebuild {:>10}",
                metric_name(ruleset),
                metric_duration(search),
                metric_duration(merge),
                metric_duration(rebuild)
            );
        }

        let rules = sorted_rule_metrics(&report);
        for (rule, elapsed, matches) in rules
            .iter()
            .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO || *matches > 0)
            .take(5)
        {
            eprintln!(
                "      rule    {:<96} {:>10} | matches {}",
                metric_name(rule),
                metric_duration(*elapsed),
                matches
            );
        }
        if elapsed >= SLOW_PHASE_TIME || tuple_delta.abs() >= BIG_TUPLE_DELTA {
            print_slow_phase_detail(phase, &report, tuple_delta, elapsed, &rules);
        }
    }

    phases.push(EgglogPhaseReport {
        name: phase.name.clone(),
        schedule: phase.schedule.clone(),
        updated,
        iterations,
        tuples_before,
        tuples_after,
        total_time: elapsed,
    });

    Ok(updated)
}
