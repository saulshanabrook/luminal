//! Static CPU rewrite rules.

use super::schema::*;
use egglog_experimental::typed::{builtins as egg, prelude::*};

// Repeated operands and multiplication by zero author symbolic rule patterns;
// neither evaluates host arithmetic or erases an expression during construction.
#[allow(clippy::eq_op, clippy::erasing_op)]
#[ruleset]
pub(crate) fn expression(
    v_rw_dim: &DimExpr,
    va_dim: &DimExpr,
    vb_dim: &DimExpr,
    ve_dim: &DimExpr,
    va_num: &egg::I64,
    vb_num: &egg::I64,
    vans: &egg::I64,
    vprod: &egg::I64,
    vexpr: &DimExpr,
    vx: &DimExpr,
    vn_num: &egg::I64,
    vr: &egg::I64,
    vy_dim: &DimExpr,
    vy_num: &egg::I64,
    vz_num: &egg::I64,
    vz_dim: &DimExpr,
    vc_num: &egg::I64,
    vc_dim: &DimExpr,
    vv: &egg::String,
    vn_dim: &DimExpr,
    vz_name: &egg::String,
    vfind: &DimExpr,
    vreplace: &DimExpr,
    ve_list: &EList,
    vlist: &EList,
    vprev_len: &egg::I64,
    vlist_len: &egg::I64,
    vother_nth: &DimExpr,
    vdim: &DimExpr,
    vother: &EList,
    vother_elems: &DimExpr,
    vn_elems: &DimExpr,
    vfrom: &DimExpr,
    vto: &DimExpr,
    vind: &egg::I64,
) -> Vec<Rule> {
    vec![
        rewrite(va_dim * vb_dim, vb_dim * va_dim).label("mul-comm"),
        rewrite(va_dim + vb_dim, vb_dim + va_dim).label("add-comm"),
        rule(
            (
                eq(ve_dim, DimExpr::MNum(va_num) + DimExpr::MNum(vb_num)),
                eq(vans, va_num + vb_num),
            ),
            (union(ve_dim, vans), subsume(DimExpr::MNum(va_num) + vb_num)),
        ),
        rewrite(
            DimExpr::MNum(va_num) - DimExpr::MNum(vb_num),
            va_num - vb_num,
        )
        .label("sub-const"),
        rule(
            (
                eq(ve_dim, DimExpr::MNum(va_num) * DimExpr::MNum(vb_num)),
                eq(vprod, va_num * vb_num),
            ),
            (
                union(ve_dim, vprod),
                subsume(DimExpr::MNum(va_num) * vb_num),
            ),
        ),
        rule(
            (
                eq(vexpr, (vx * DimExpr::MNum(va_num)) * DimExpr::MNum(vb_num)),
                eq(vprod, va_num * vb_num),
            ),
            (union(vexpr, vx * vprod),),
        )
        .label("fold-right-associated-const-mul"),
        rule(
            (
                eq(vexpr, DimExpr::MNum(vb_num) * (vx * DimExpr::MNum(va_num))),
                eq(vprod, va_num * vb_num),
            ),
            (union(vexpr, vx * vprod),),
        )
        .label("fold-left-associated-const-mul"),
        rule(
            (
                eq(v_rw_dim, DimExpr::MNum(va_num) / DimExpr::MNum(vb_num)),
                ne(egg::I64::from(0i64), vb_num),
            ),
            (union(v_rw_dim, va_num / vb_num),),
        )
        .label("div-const"),
        rewrite(va_dim / va_dim, 1i64).label("div-self"),
        rule(
            (
                eq(
                    v_rw_dim,
                    (vx * DimExpr::MNum(vn_num)) / DimExpr::MNum(vn_num),
                ),
                vn_num.ge(1i64),
            ),
            (union(v_rw_dim, vx),),
        )
        .label("div-mul-num-self"),
        rule(
            (
                eq(
                    v_rw_dim,
                    ((vx * DimExpr::MNum(vn_num)) + DimExpr::MNum(vr)) / DimExpr::MNum(vn_num),
                ),
                vn_num.ge(1i64),
                vr.ge(0i64),
                vr.lt(vn_num),
            ),
            (union(v_rw_dim, vx),),
        )
        .label("div-mul-num-plus-rem"),
        rule(
            (
                eq(
                    v_rw_dim,
                    DimExpr::MCeilDiv(DimExpr::MNum(va_num), DimExpr::MNum(vb_num)),
                ),
                ne(egg::I64::from(0i64), vb_num),
                eq(egg::I64::from(0i64), va_num % vb_num),
            ),
            (union(v_rw_dim, va_num / vb_num),),
        )
        .label("ceildiv-const"),
        rewrite(
            DimExpr::MMax(DimExpr::MNum(va_num), DimExpr::MNum(vb_num)),
            va_num.max(vb_num),
        )
        .label("max-const"),
        rewrite(
            DimExpr::MMin(DimExpr::MNum(va_num), DimExpr::MNum(vb_num)),
            va_num.min(vb_num),
        )
        .label("min-const"),
        rewrite(
            DimExpr::MAnd(DimExpr::MNum(va_num), DimExpr::MNum(vb_num)),
            va_num & vb_num,
        )
        .label("and-const"),
        rewrite(DimExpr::MFloat(-1.0), -1i64).label("float-neg1-to-num"),
        rewrite(DimExpr::MNum(-1i64), -1.0).label("num-neg1-to-float"),
        rewrite(va_dim + 0i64, va_dim).label("add-zero"),
        rewrite(va_dim * 1i64, va_dim),
        rule(
            (eq(ve_dim, va_dim * 0i64),),
            (union(ve_dim, 0i64), subsume(va_dim * 0i64)),
        ),
        rewrite(va_dim / 1i64, va_dim).label("div-one"),
        rewrite((vx * vy_dim) % vy_dim, 0i64).label("mod-mul-self"),
        rule(
            (
                eq(v_rw_dim, DimExpr::MNum(va_num) % DimExpr::MNum(vb_num)),
                ne(egg::I64::from(0i64), vb_num),
            ),
            (union(v_rw_dim, va_num % vb_num),),
        )
        .label("mod-const"),
        rule(
            (
                eq(
                    v_rw_dim,
                    ((vx * DimExpr::MNum(vn_num)) + DimExpr::MNum(vr)) % DimExpr::MNum(vn_num),
                ),
                vn_num.ge(1i64),
                vr.ge(0i64),
                vr.lt(vn_num),
            ),
            (union(v_rw_dim, vr),),
        )
        .label("mod-mul-num-plus-rem"),
        rule(
            (
                eq(
                    v_rw_dim,
                    (vx % DimExpr::MNum(vy_num)) % DimExpr::MNum(vz_num),
                ),
                vz_num.ge(vy_num),
                eq(egg::I64::from(0i64), vy_num % vz_num),
            ),
            (union(v_rw_dim, vx % vy_num),),
        )
        .label("mod-mod-larger"),
        rule(
            (
                eq(
                    v_rw_dim,
                    (vx % DimExpr::MNum(vy_num)) % DimExpr::MNum(vz_num),
                ),
                vy_num.ge(vz_num),
                eq(egg::I64::from(0i64), vz_num % vy_num),
            ),
            (union(v_rw_dim, vx % vz_num),),
        )
        .label("mod-mod-smaller"),
        rewrite(((vz_dim / vx) * vx) + (vz_dim % vx), vz_dim).label("merge-dims"),
        rule(
            (
                eq(
                    v_rw_dim,
                    (va_dim / DimExpr::MNum(vb_num)) / DimExpr::MNum(vc_num),
                ),
                vb_num.ge(1i64),
                vc_num.ge(1i64),
                vb_num.lt(3037000500i64),
                vc_num.lt(3037000500i64),
            ),
            (union(v_rw_dim, va_dim / (vb_num * vc_num)),),
        )
        .label("div-div-num"),
        rewrite(
            (va_dim / vb_dim) + vc_dim,
            (va_dim + (vc_dim * vb_dim)) / vb_dim,
        )
        .label("add-div"),
        rewrite(va_dim + (vb_dim - va_dim), vb_dim).label("add-sub-cancel"),
        rewrite((vb_dim - va_dim) + va_dim, vb_dim).label("add-sub-cancel2"),
        rewrite(va_dim - va_dim, 0i64).label("sub-self"),
        rewrite(
            (va_dim - DimExpr::MNum(vb_num)) + DimExpr::MNum(vc_num),
            va_dim - (vb_num - vc_num),
        )
        .label("add-sub-const"),
        rewrite(
            DimExpr::MNum(vc_num) + (va_dim - DimExpr::MNum(vb_num)),
            va_dim - (vb_num - vc_num),
        )
        .label("add-sub-const2"),
        rewrite(
            (va_dim + DimExpr::MNum(vb_num)) - DimExpr::MNum(vc_num),
            va_dim + (vb_num - vc_num),
        )
        .label("sub-add-const"),
        rewrite(
            (va_dim - DimExpr::MNum(vb_num)) - DimExpr::MNum(vc_num),
            va_dim - (vb_num + vc_num),
        )
        .label("sub-sub-const"),
        rewrite(
            (va_dim * vb_dim) + (va_dim * vc_dim),
            va_dim * (vb_dim + vc_dim),
        )
        .label("factor"),
        rewrite(va_dim + va_dim, DimExpr::MNum(2i64) * va_dim).label("double"),
        rule(
            (
                eq(
                    ve_dim,
                    (va_dim + DimExpr::MNum(vb_num)) + DimExpr::MNum(vc_num),
                ),
                eq(vans, vb_num + vc_num),
            ),
            (
                union(ve_dim, va_dim + vans),
                subsume((va_dim + vb_num) + vc_num),
            ),
        ),
        rewrite(
            (DimExpr::MNum(vb_num) + DimExpr::MVar(vv)) + DimExpr::MNum(vc_num),
            DimExpr::MVar(vv) + (vb_num + vc_num),
        )
        .label("add-assoc-var"),
        rewrite(
            (DimExpr::MNum(vb_num) + (vn_dim * va_dim)) + DimExpr::MNum(vc_num),
            (vn_dim * va_dim) + (vb_num + vc_num),
        )
        .label("add-assoc-mul"),
        rule(
            (eq(v_rw_dim, (DimExpr::MNum(vn_num) * va_dim) + va_dim),),
            (
                union(v_rw_dim, DimExpr::MNum(vn_num + 1i64) * va_dim),
                subsume((DimExpr::MNum(vn_num) * va_dim) + va_dim),
            ),
        )
        .label("combine-like-1"),
        rule(
            (eq(v_rw_dim, va_dim + (DimExpr::MNum(vn_num) * va_dim)),),
            (
                union(v_rw_dim, DimExpr::MNum(vn_num + 1i64) * va_dim),
                subsume(va_dim + (DimExpr::MNum(vn_num) * va_dim)),
            ),
        )
        .label("combine-like-2"),
        rule(
            (eq(v_rw_dim, (va_dim * DimExpr::MNum(vn_num)) + va_dim),),
            (
                union(v_rw_dim, DimExpr::MNum(vn_num + 1i64) * va_dim),
                subsume((va_dim * vn_num) + va_dim),
            ),
        )
        .label("combine-like-3"),
        rule(
            (eq(v_rw_dim, va_dim + (va_dim * DimExpr::MNum(vn_num))),),
            (
                union(v_rw_dim, DimExpr::MNum(vn_num + 1i64) * va_dim),
                subsume(va_dim + (va_dim * vn_num)),
            ),
        )
        .label("combine-like-4"),
        rule(
            (eq(
                v_rw_dim,
                (va_dim + DimExpr::MVar(vv)) + DimExpr::MVar(vv),
            ),),
            (
                union(v_rw_dim, va_dim + (DimExpr::MNum(2i64) * DimExpr::MVar(vv))),
                subsume((va_dim + DimExpr::MVar(vv)) + DimExpr::MVar(vv)),
            ),
        )
        .label("combine-var-1"),
        rule(
            (eq(
                v_rw_dim,
                (DimExpr::MVar(vv) + va_dim) + DimExpr::MVar(vv),
            ),),
            (
                union(v_rw_dim, va_dim + (DimExpr::MNum(2i64) * DimExpr::MVar(vv))),
                subsume((DimExpr::MVar(vv) + va_dim) + DimExpr::MVar(vv)),
            ),
        )
        .label("combine-var-2"),
        rule(
            (eq(
                v_rw_dim,
                ((DimExpr::MNum(vn_num) * va_dim) + vb_dim) + va_dim,
            ),),
            (
                union(v_rw_dim, (DimExpr::MNum(vn_num + 1i64) * va_dim) + vb_dim),
                subsume(((DimExpr::MNum(vn_num) * va_dim) + vb_dim) + va_dim),
            ),
        )
        .label("accum-1"),
        rule(
            (eq(
                v_rw_dim,
                (vb_dim + (DimExpr::MNum(vn_num) * va_dim)) + va_dim,
            ),),
            (
                union(v_rw_dim, vb_dim + (DimExpr::MNum(vn_num + 1i64) * va_dim)),
                subsume((vb_dim + (DimExpr::MNum(vn_num) * va_dim)) + va_dim),
            ),
        )
        .label("accum-2"),
        rule(
            (
                eq(v_rw_dim, DimExpr::MReplace(vx, vy_dim, vz_dim)),
                eq(vx, vy_dim),
            ),
            (union(v_rw_dim, vz_dim),),
        )
        .label("replace-match"),
        rewrite(
            DimExpr::MReplace(va_dim + vb_dim, vx, vy_dim),
            DimExpr::MReplace(va_dim, vx, vy_dim) + DimExpr::MReplace(vb_dim, vx, vy_dim),
        )
        .label("replace-MAdd"),
        rewrite(
            DimExpr::MReplace(va_dim - vb_dim, vx, vy_dim),
            DimExpr::MReplace(va_dim, vx, vy_dim) - DimExpr::MReplace(vb_dim, vx, vy_dim),
        )
        .label("replace-MSub"),
        rewrite(
            DimExpr::MReplace(va_dim * vb_dim, vx, vy_dim),
            DimExpr::MReplace(va_dim, vx, vy_dim) * DimExpr::MReplace(vb_dim, vx, vy_dim),
        )
        .label("replace-MMul"),
        rewrite(
            DimExpr::MReplace(va_dim / vb_dim, vx, vy_dim),
            DimExpr::MReplace(va_dim, vx, vy_dim) / DimExpr::MReplace(vb_dim, vx, vy_dim),
        )
        .label("replace-MDiv"),
        rewrite(
            DimExpr::MReplace(DimExpr::MCeilDiv(va_dim, vb_dim), vx, vy_dim),
            DimExpr::MCeilDiv(
                DimExpr::MReplace(va_dim, vx, vy_dim),
                DimExpr::MReplace(vb_dim, vx, vy_dim),
            ),
        )
        .label("replace-MCeilDiv"),
        rewrite(
            DimExpr::MReplace(va_dim % vb_dim, vx, vy_dim),
            DimExpr::MReplace(va_dim, vx, vy_dim) % DimExpr::MReplace(vb_dim, vx, vy_dim),
        )
        .label("replace-MMod"),
        rewrite(
            DimExpr::MReplace(DimExpr::MMin(va_dim, vb_dim), vx, vy_dim),
            DimExpr::MMin(
                DimExpr::MReplace(va_dim, vx, vy_dim),
                DimExpr::MReplace(vb_dim, vx, vy_dim),
            ),
        )
        .label("replace-MMin"),
        rewrite(
            DimExpr::MReplace(DimExpr::MMax(va_dim, vb_dim), vx, vy_dim),
            DimExpr::MMax(
                DimExpr::MReplace(va_dim, vx, vy_dim),
                DimExpr::MReplace(vb_dim, vx, vy_dim),
            ),
        )
        .label("replace-MMax"),
        rewrite(
            DimExpr::MReplace(DimExpr::MFloorTo(va_dim, vb_dim), vx, vy_dim),
            DimExpr::MFloorTo(
                DimExpr::MReplace(va_dim, vx, vy_dim),
                DimExpr::MReplace(vb_dim, vx, vy_dim),
            ),
        )
        .label("replace-MFloorTo"),
        rewrite(DimExpr::MReplace(DimExpr::MNum(vn_num), vx, vy_dim), vn_num).label("replace-num"),
        rule(
            (
                eq(
                    v_rw_dim,
                    DimExpr::MReplace(DimExpr::MVar(vz_name), vfind, vreplace),
                ),
                ne(vfind, DimExpr::MVar(vz_name)),
            ),
            (union(v_rw_dim, DimExpr::MVar(vz_name)),),
        )
        .label("replace-var-miss"),
        rule(
            (
                eq(
                    v_rw_dim,
                    DimExpr::MReplace(DimExpr::MIter(), vfind, vreplace),
                ),
                ne(vfind, DimExpr::MIter()),
            ),
            (union(v_rw_dim, DimExpr::MIter()),),
        )
        .label("replace-iter-miss"),
        // Expression lists.
        rule((eq(ve_list, EList::ENil()),), (set(ve_list.len(), 0i64),)),
        rule(
            (
                eq(ve_list, EList::ECons(vexpr, vlist)),
                eq(vprev_len, vlist.len()),
            ),
            (set(ve_list.len(), vprev_len + 1i64),),
        ),
        rule(
            (
                eq(ve_list, EList::ECons(vexpr, vlist)),
                eq(vlist_len, vlist.len()),
            ),
            (set(ve_list.nth_from_end(vlist_len), vexpr),),
        ),
        rule(
            (
                eq(ve_list, EList::ECons(vexpr, vlist)),
                eq(vother_nth, vlist.nth_from_end(vn_num)),
            ),
            (set(ve_list.nth_from_end(vn_num), vother_nth),),
        ),
        rule(
            (eq(ve_list, EList::ENil()),),
            (set(ve_list.n_elements(), 1i64),),
        ),
        rule(
            (
                eq(ve_list, EList::ECons(vdim, vother)),
                eq(vother_elems, vother.n_elements()),
            ),
            (set(ve_list.n_elements(), vdim * vother_elems),),
        ),
        rule(
            (
                eq(vother, EConsArgs::fresh()),
                eq(
                    vlist,
                    EConsArgs {
                        tail: vother.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                eq(ve_list, EList::RowMajor(vlist)),
                eq(vn_elems, vother.n_elements()),
            ),
            (union(
                ve_list,
                EList::ECons(vn_elems * DimExpr::MIter(), EList::RowMajor(vother)),
            ),),
        ),
        rewrite(EList::RowMajor([vdim]), [DimExpr::MIter()]).label("rowmajor-base"),
        rewrite(
            EList::MReplaceList(EList::ECons(vexpr, vlist), vfrom, vto),
            EList::ECons(
                DimExpr::MReplace(vexpr, vfrom, vto),
                EList::MReplaceList(vlist, vfrom, vto),
            ),
        )
        .label("replace-list-cons"),
        rule(
            (
                eq(
                    ve_list,
                    EList::ReplaceNthFromEnd(EList::ECons(vexpr, vlist), vto, vind),
                ),
                eq(vind, vlist.len()),
            ),
            (union(ve_list, EList::ECons(vto, vlist)),),
        ),
        rule(
            (
                eq(
                    ve_list,
                    EList::ReplaceNthFromEnd(EList::ECons(vexpr, vlist), vto, vind),
                ),
                vind.lt(vlist.len()),
            ),
            (union(
                ve_list,
                EList::ECons(vexpr, EList::ReplaceNthFromEnd(vlist, vto, vind)),
            ),),
        ),
        rule(
            (
                eq(
                    ve_list,
                    EList::RemoveNthFromEnd(EList::ECons(vexpr, vlist), vind),
                ),
                eq(vind, vlist.len()),
            ),
            (union(ve_list, vlist),),
        ),
        rule(
            (
                eq(
                    ve_list,
                    EList::RemoveNthFromEnd(EList::ECons(vexpr, vlist), vind),
                ),
                vind.lt(vlist.len()),
            ),
            (union(
                ve_list,
                EList::ECons(vexpr, EList::RemoveNthFromEnd(vlist, vind)),
            ),),
        ),
    ]
}

#[ruleset]
pub(crate) fn intervals(
    ve: &DimExpr,
    vn: &egg::I64,
    va: &DimExpr,
    vb: &DimExpr,
    vlo_a: &egg::I64,
    vlo_b: &egg::I64,
    vsum: &egg::I64,
    vhi_a: &egg::I64,
    vhi_b: &egg::I64,
    v_rw: &DimExpr,
    vx: &DimExpr,
    vhi: &egg::I64,
    vlo: &egg::I64,
) -> Vec<Rule> {
    vec![
        rule(
            (eq(ve, DimExpr::MNum(vn)),),
            (set(ve.lower(), vn), set(ve.upper(), vn)),
        )
        .label("interval-num-exact"),
        rule(
            (
                eq(ve, va + vb),
                eq(vlo_a, va.lower()),
                eq(vlo_b, vb.lower()),
                eq(vsum, vlo_a + vlo_b),
                vlo_a.ge(0i64),
                vlo_b.ge(0i64),
                (egg::I64::from(9223372036854775807i64) - vlo_b).ge(vlo_a),
            ),
            (set(ve.lower(), vsum),),
        )
        .label("interval-add-lower-nonnegative"),
        rule(
            (
                eq(ve, va + vb),
                eq(vhi_a, va.upper()),
                eq(vhi_b, vb.upper()),
                eq(vsum, vhi_a + vhi_b),
                vhi_a.lt(9223372036854775807i64),
                vhi_b.lt(9223372036854775807i64),
                (egg::I64::from(9223372036854775807i64) - vhi_b).ge(vhi_a),
            ),
            (set(ve.upper(), vsum),),
        )
        .label("interval-add-upper-finite"),
        rule(
            (
                eq(ve, DimExpr::MMin(va, vb)),
                eq(vlo_a, va.lower()),
                eq(vlo_b, vb.lower()),
            ),
            (set(ve.lower(), vlo_a.min(vlo_b)),),
        )
        .label("interval-min-lower"),
        rule(
            (
                eq(ve, DimExpr::MMin(va, vb)),
                eq(vhi_a, va.upper()),
                eq(vhi_b, vb.upper()),
            ),
            (set(ve.upper(), vhi_a.min(vhi_b)),),
        )
        .label("interval-min-upper"),
        rule(
            (
                eq(ve, DimExpr::MMax(va, vb)),
                eq(vlo_a, va.lower()),
                eq(vlo_b, vb.lower()),
            ),
            (set(ve.lower(), vlo_a.max(vlo_b)),),
        )
        .label("interval-max-lower"),
        rule(
            (
                eq(ve, DimExpr::MMax(va, vb)),
                eq(vhi_a, va.upper()),
                eq(vhi_b, vb.upper()),
            ),
            (set(ve.upper(), vhi_a.max(vhi_b)),),
        )
        .label("interval-max-upper"),
        rule(
            (
                eq(v_rw, DimExpr::MLt(vx, DimExpr::MNum(vn))),
                eq(vhi, vx.upper()),
                vhi.lt(vn),
            ),
            (union(v_rw, 1i64),),
        )
        .label("interval-lt-true"),
        rule(
            (
                eq(v_rw, DimExpr::MLt(vx, DimExpr::MNum(vn))),
                eq(vlo, vx.lower()),
                vlo.ge(vn),
            ),
            (union(v_rw, 0i64),),
        )
        .label("interval-lt-false"),
        rule(
            (
                eq(v_rw, DimExpr::MGte(vx, DimExpr::MNum(vn))),
                eq(vlo, vx.lower()),
                vlo.ge(vn),
            ),
            (union(v_rw, 1i64),),
        )
        .label("interval-gte-true"),
        rule(
            (
                eq(v_rw, DimExpr::MGte(vx, DimExpr::MNum(vn))),
                eq(vhi, vx.upper()),
                vhi.lt(vn),
            ),
            (union(v_rw, 0i64),),
        )
        .label("interval-gte-false"),
        rule(
            (
                eq(v_rw, DimExpr::MMin(vx, DimExpr::MNum(vn))),
                eq(vhi, vx.upper()),
                vn.ge(vhi),
            ),
            (union(v_rw, vx),),
        )
        .label("interval-min-right-identity"),
        rule(
            (
                eq(v_rw, DimExpr::MMax(vx, DimExpr::MNum(vn))),
                eq(vlo, vx.lower()),
                vlo.ge(vn),
            ),
            (union(v_rw, vx),),
        )
        .label("interval-max-right-identity"),
        rule(
            (
                eq(v_rw, vx % DimExpr::MNum(vn)),
                vn.ge(1i64),
                eq(vlo, vx.lower()),
                eq(vhi, vx.upper()),
                vlo.ge(0i64),
                vhi.lt(vn),
            ),
            (union(v_rw, vx),),
        )
        .label("interval-mod-small"),
        rule(
            (
                eq(v_rw, vx / DimExpr::MNum(vn)),
                vn.ge(1i64),
                eq(vlo, vx.lower()),
                eq(vhi, vx.upper()),
                vlo.ge(0i64),
                vhi.lt(vn),
            ),
            (union(v_rw, 0i64),),
        )
        .label("interval-div-small"),
    ]
}

#[ruleset]
pub(crate) fn cleanup(
    va_dim: &DimExpr,
    vb_dim: &DimExpr,
    vc_dim: &DimExpr,
    va_list: &EList,
    vc_num: &egg::I64,
    vb_num: &egg::I64,
    vx: &EList,
    vy: &egg::I64,
) -> Vec<Rule> {
    vec![
        rule(
            (DimExpr::MReplace(va_dim, vb_dim, vc_dim),),
            (delete(DimExpr::MReplace(va_dim, vb_dim, vc_dim)),),
        ),
        rule(
            (EList::MReplaceList(va_list, vb_dim, vc_dim),),
            (delete(EList::MReplaceList(va_list, vb_dim, vc_dim)),),
        ),
        rule(
            (EList::ReplaceNthFromEnd(va_list, vb_dim, vc_num),),
            (delete(EList::ReplaceNthFromEnd(va_list, vb_dim, vc_num)),),
        ),
        rule(
            (EList::RemoveNthFromEnd(va_list, vb_num),),
            (delete(EList::RemoveNthFromEnd(va_list, vb_num)),),
        ),
        rule((EList::RowMajor(vx),), (delete(EList::RowMajor(vx)),)),
        rule((vx.len(),), (delete(vx.len()),)),
        rule((vx.nth_from_end(vy),), (delete(vx.nth_from_end(vy)),)),
        rule((vx.n_elements(),), (delete(vx.n_elements()),)),
    ]
}

#[ruleset]
pub(crate) fn matmul() -> Ruleset {
    ruleset((
        &crate::egglog_utils::matmul_flattening::squeeze::matmul_squeeze,
        &crate::egglog_utils::matmul_flattening::batch_merge_a_contig::matmul_merge_a_contiguous,
        &crate::egglog_utils::matmul_flattening::batch_merge_b_contig::matmul_merge_b_contiguous,
    ))
}

#[ruleset]
pub(crate) fn loops(
    vl: &IList,
    vx: &Ir,
    vtail: &IList,
    ve: &Ir,
    vid: &egg::I64,
    vstream: &egg::I64,
    vdt: &DType,
    vcont: &IList,
) -> Vec<Rule> {
    vec![
        rule((eq(vl, [vx]),), (vl.identical_inputs(),)).label("identical_inputs base"),
        rule(
            (
                eq(vl, IList::ICons(vx, IList::ICons(vx, vtail))),
                (IList::ICons(vx, vtail)).identical_inputs(),
            ),
            (vl.identical_inputs(),),
        )
        .label("identical_inputs ind"),
        rule(
            (
                eq(
                    ve,
                    Ir::Op(
                        OpKind::LoopInput(vid, vstream, vdt),
                        IList::ICons(vx, vcont),
                    ),
                ),
                (IList::ICons(vx, vcont)).identical_inputs(),
            ),
            (union(
                ve,
                Ir::Op(OpKind::LoopInputStatic(vid, vstream, vdt), [vx]),
            ),),
        )
        .label("LoopInput to LoopInputStatic"),
        rewrite(Ir::Op(LoopInputStaticArgs::fresh(), [vx]), vx).label("LoopInputStatic inline"),
    ]
}
