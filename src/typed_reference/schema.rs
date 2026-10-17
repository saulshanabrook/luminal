//! Typed declarations for the complete ReferenceRuntime authoring catalog.
//! Callable identities are shared by core and backend rules.

use egglog_experimental::typed::{builtins as egg, prelude::*};

#[sort(name = "Expression")]
pub struct DimExpr;

#[declarations]
impl DimExpr {
    #[constructor(name = "MNum", from(i32, i64, &i32, &i64))]
    pub fn MNum(n: egg::I64) -> Self;
    #[constructor(name = "MFloat", from(f64, &f64))]
    pub fn MFloat(value: egg::F64) -> Self;
    #[constructor(name = "MIter")]
    pub fn MIter() -> Self;
    #[constructor(name = "MVar", from(&str, std::string::String, &std::string::String))]
    pub fn MVar(name: egg::String) -> Self;
    #[constructor(name = "MCeilDiv")]
    pub fn MCeilDiv(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MMin")]
    pub fn MMin(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MMax")]
    pub fn MMax(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MAnd")]
    pub fn MAnd(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MOr")]
    pub fn MOr(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MGte")]
    pub fn MGte(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MLt")]
    pub fn MLt(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MFloorTo")]
    pub fn MFloorTo(a: DimExpr, b: DimExpr) -> Self;
    #[constructor(name = "MReplace")]
    pub fn MReplace(value: DimExpr, from: DimExpr, to: DimExpr) -> Self;
}

// These conversions author constructor nodes; they never evaluate or detach
// their inputs. In particular, names are MVar data, not rule variables.
impl From<char> for DimExpr {
    fn from(value: char) -> Self {
        Self::MVar(value.to_string())
    }
}
impl From<&char> for DimExpr {
    fn from(value: &char) -> Self {
        Self::MVar(value.to_string())
    }
}

#[sort(name = "EList")]
pub struct EList;

#[declarations]
impl EList {
    #[constructor(name = "ECons", args = EConsArgs)]
    pub fn ECons(head: DimExpr, tail: EList) -> Self;
    #[constructor(name = "ENil")]
    pub fn ENil() -> Self;
    #[constructor(name = "MReplaceList")]
    pub fn MReplaceList(value: EList, from: DimExpr, to: DimExpr) -> Self;
    #[constructor(name = "ReplaceNthFromEnd")]
    pub fn ReplaceNthFromEnd(value: EList, to: DimExpr, index: egg::I64) -> Self;
    #[constructor(name = "RemoveNthFromEnd")]
    pub fn RemoveNthFromEnd(value: EList, index: egg::I64) -> Self;
    #[constructor(name = "RowMajor")]
    pub fn RowMajor(value: EList) -> Self;
}

#[sort(name = "DType")]
pub struct DType;

#[declarations]
impl DType {
    #[constructor(name = "F32")]
    pub fn F32() -> Self;
    #[constructor(name = "F64")]
    pub fn F64() -> Self;
    #[constructor(name = "F16")]
    pub fn F16() -> Self;
    #[constructor(name = "Bf16")]
    pub fn Bf16() -> Self;
    #[constructor(name = "Int")]
    pub fn Int() -> Self;
    #[constructor(name = "Int64")]
    pub fn Int64() -> Self;
    #[constructor(name = "Bool")]
    pub fn Bool() -> Self;
    #[constructor(name = "F4E2M1")]
    pub fn F4E2M1() -> Self;
    #[constructor(name = "F8E4M3")]
    pub fn F8E4M3() -> Self;
    #[constructor(name = "F8E5M2")]
    pub fn F8E5M2() -> Self;
    #[constructor(name = "F8UE8M0")]
    pub fn F8UE8M0() -> Self;
    #[constructor(name = "I4")]
    pub fn I4() -> Self;
    #[constructor(name = "U4")]
    pub fn U4() -> Self;
    #[constructor(name = "I8")]
    pub fn I8() -> Self;
    #[constructor(name = "U8")]
    pub fn U8() -> Self;
    #[constructor(name = "I16")]
    pub fn I16() -> Self;
    #[constructor(name = "U16")]
    pub fn U16() -> Self;
    #[constructor(name = "TF32")]
    pub fn TF32() -> Self;
    #[constructor(name = "F6E2M3")]
    pub fn F6E2M3() -> Self;
    #[constructor(name = "F6E3M2")]
    pub fn F6E3M2() -> Self;
}

impl From<crate::dtype::DType> for DType {
    fn from(value: crate::dtype::DType) -> Self {
        use crate::dtype::DType as Host;
        match value {
            Host::F32 => Self::F32(),
            Host::F64 => Self::F64(),
            Host::F16 => Self::F16(),
            Host::Bf16 => Self::Bf16(),
            Host::TF32 => Self::TF32(),
            Host::Int => Self::Int(),
            Host::I64 => Self::Int64(),
            Host::I4 => Self::I4(),
            Host::U4 => Self::U4(),
            Host::I8 => Self::I8(),
            Host::U8 => Self::U8(),
            Host::I16 => Self::I16(),
            Host::U16 => Self::U16(),
            Host::Bool => Self::Bool(),
            Host::F8UE8M0 => Self::F8UE8M0(),
            Host::F8E4M3 => Self::F8E4M3(),
            Host::F8E5M2 => Self::F8E5M2(),
            Host::F6E2M3 => Self::F6E2M3(),
            Host::F6E3M2 => Self::F6E3M2(),
            Host::F4E2M1 => Self::F4E2M1(),
        }
    }
}
impl From<&crate::dtype::DType> for DType {
    fn from(value: &crate::dtype::DType) -> Self {
        Self::from(*value)
    }
}

/// Read a concrete dtype constructor without evaluating or selecting a term.
/// Frozen equality classes must first be resolved to an explicit producer.
impl TryFrom<&DType> for crate::dtype::DType {
    type Error = TypedError;

    fn try_from(value: &DType) -> Result<Self, Self::Error> {
        macro_rules! dtype {
            ($($ctor:ident => $host:ident),* $(,)?) => { $(
                if get_args(value, DType::$ctor)?.is_some() { return Ok(Self::$host); }
            )* };
        }
        dtype!(F32=>F32,F64=>F64,F16=>F16,Bf16=>Bf16,Int=>Int,Int64=>I64,Bool=>Bool,F4E2M1=>F4E2M1,F6E2M3=>F6E2M3,F6E3M2=>F6E3M2,F8E4M3=>F8E4M3,F8E5M2=>F8E5M2,F8UE8M0=>F8UE8M0,I4=>I4,U4=>U4,I8=>I8,U8=>U8,I16=>I16,U16=>U16,TF32=>TF32);
        Err(TypedError::Decode(
            "expected a concrete dtype constructor".into(),
        ))
    }
}

impl TryFrom<DType> for crate::dtype::DType {
    type Error = TypedError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

#[sort(name = "IR")]
pub struct Ir;

#[declarations]
impl Ir {
    #[constructor(name = "Input", args = InputArgs)]
    pub fn Input(node: egg::I64, label: egg::String, dtype: DType) -> Self;
    #[constructor(name = "Output", args = OutputArgs)]
    pub fn Output(inp: Ir, node: egg::I64) -> Self;
    #[constructor(name = "LoopStart", args = LoopStartArgs)]
    pub fn LoopStart(
        inp: Ir,
        loop_id: egg::I64,
        slot_idx: egg::I64,
        iters: DimExpr,
        dtype: DType,
    ) -> Self;
    #[constructor(name = "LoopEnd", args = LoopEndArgs)]
    pub fn LoopEnd(inp: Ir, loop_id: egg::I64, slot_idx: egg::I64, dtype: DType) -> Self;
    #[constructor(name = "Op", args = OpArgs)]
    pub fn Op(kind: OpKind, inputs: IList) -> Self;
}

#[sort(name = "IList")]
pub struct IList;

#[declarations]
impl IList {
    #[constructor(name = "ICons")]
    pub fn ICons(head: Ir, tail: IList) -> Self;
    #[constructor(name = "INil")]
    pub fn INil() -> Self;
}

#[constructor(name = "ConsumedBuffer")]
pub fn consumed_buffer(input: Ir) -> Ir;

macro_rules! ordered_list_from {
    ($list:ident, $element:ty, $cons:ident, $nil:ident) => {
        impl $list {
            // Build from the end without recursion, retaining original element
            // order, duplicates, and any authored or observed shared children.
            fn from_ordered_elements(elements: impl DoubleEndedIterator<Item = $element>) -> Self {
                elements
                    .rev()
                    .fold(Self::$nil(), |tail, head| Self::$cons(head, tail))
            }
        }
        impl<T: Into<$element>> FromIterator<T> for $list {
            fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
                // Convert in source order before reversing the cons construction.
                let elements: Vec<$element> = values.into_iter().map(Into::into).collect();
                Self::from_ordered_elements(elements.into_iter())
            }
        }
        impl<T: Into<$element>, const N: usize> From<[T; N]> for $list {
            fn from(values: [T; N]) -> Self {
                Self::from_ordered_elements(values.map(Into::into).into_iter())
            }
        }
        impl<T: Into<$element>> From<Vec<T>> for $list {
            fn from(values: Vec<T>) -> Self {
                values.into_iter().collect()
            }
        }
        impl<T: Clone + Into<$element>> From<&[T]> for $list {
            fn from(values: &[T]) -> Self {
                values.iter().cloned().collect()
            }
        }
        impl<T: Clone + Into<$element>, const N: usize> From<&[T; N]> for $list {
            fn from(values: &[T; N]) -> Self {
                values.iter().cloned().collect()
            }
        }
        impl<T: Clone + Into<$element>> From<&Vec<T>> for $list {
            fn from(values: &Vec<T>) -> Self {
                values.iter().cloned().collect()
            }
        }
    };
}
ordered_list_from!(EList, DimExpr, ECons, ENil);
ordered_list_from!(IList, Ir, ICons, INil);

#[sort(name = "OpKind")]
pub struct OpKind;

#[declarations]
impl OpKind {
    #[constructor(name = "CustomOpKind", args = CustomOpKindArgs)]
    pub fn CustomOpKind(id: egg::I64, dtype: DType) -> Self;
    #[constructor(name = "LoopInput", args = LoopInputArgs)]
    pub fn LoopInput(loop_id: egg::I64, stream_id: egg::I64, dtype: DType) -> Self;
    #[constructor(name = "LoopInputStatic", args = LoopInputStaticArgs)]
    pub fn LoopInputStatic(loop_id: egg::I64, stream_id: egg::I64, dtype: DType) -> Self;
    #[constructor(name = "LoopOutput", args = LoopOutputArgs)]
    pub fn LoopOutput(loop_id: egg::I64, stream_id: egg::I64, dtype: DType) -> Self;
    #[constructor(name = "LoopOutputSelect", args = LoopOutputSelectArgs)]
    pub fn LoopOutputSelect(
        loop_id: egg::I64,
        stream_id: egg::I64,
        iter: egg::I64,
        dtype: DType,
    ) -> Self;
    #[constructor(name = "Constant", args = ConstantArgs)]
    pub fn Constant(value: egg::F64) -> Self;
    #[constructor(name = "Cast", args = CastArgs)]
    pub fn Cast(size: DimExpr, dtype: DType) -> Self;
    #[constructor(name = "Iota", args = IotaArgs)]
    pub fn Iota(expr: DimExpr, range: DimExpr) -> Self;
    #[constructor(name = "Exp2", args = Exp2Args)]
    pub fn Exp2(shape: EList, strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Log2", args = Log2Args)]
    pub fn Log2(shape: EList, strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Sin", args = SinArgs)]
    pub fn Sin(shape: EList, strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Recip", args = RecipArgs)]
    pub fn Recip(shape: EList, strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Sqrt", args = SqrtArgs)]
    pub fn Sqrt(shape: EList, strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Add", args = AddArgs)]
    pub fn Add(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Mul", args = MulArgs)]
    pub fn Mul(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Mod", args = ModArgs)]
    pub fn Mod(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "LessThan", args = LessThanArgs)]
    pub fn LessThan(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> Self;
    #[constructor(name = "Gather", args = GatherArgs)]
    pub fn Gather(
        index_shape: EList,
        index_strides: EList,
        data_shape: EList,
        data_strides: EList,
    ) -> Self;
    #[constructor(name = "Scatter", args = ScatterArgs)]
    pub fn Scatter(
        dest_shape: EList,
        dest_strides: EList,
        index_shape: EList,
        index_strides: EList,
        src_strides: EList,
    ) -> Self;
    #[constructor(name = "Sum", args = SumArgs)]
    pub fn Sum(
        shape: EList,
        iters: DimExpr,
        strides: EList,
        iter_stride: DimExpr,
        out_strides: EList,
    ) -> Self;
    #[constructor(name = "Max", args = MaxArgs)]
    pub fn Max(
        shape: EList,
        iters: DimExpr,
        strides: EList,
        iter_stride: DimExpr,
        out_strides: EList,
    ) -> Self;
}

#[declarations]
impl std::ops::Add<&DimExpr> for &DimExpr {
    type Output = DimExpr;

    #[constructor(name = "MAdd")]
    fn add(self, rhs: &DimExpr) -> DimExpr;
}

#[declarations]
impl std::ops::Sub<&DimExpr> for &DimExpr {
    type Output = DimExpr;

    #[constructor(name = "MSub")]
    fn sub(self, rhs: &DimExpr) -> DimExpr;
}

#[declarations]
impl std::ops::Mul<&DimExpr> for &DimExpr {
    type Output = DimExpr;

    #[constructor(name = "MMul")]
    fn mul(self, rhs: &DimExpr) -> DimExpr;
}

#[declarations]
impl std::ops::Div<&DimExpr> for &DimExpr {
    type Output = DimExpr;

    #[constructor(name = "MDiv")]
    fn div(self, rhs: &DimExpr) -> DimExpr;
}

#[declarations]
impl std::ops::Rem<&DimExpr> for &DimExpr {
    type Output = DimExpr;

    #[constructor(name = "MMod")]
    fn rem(self, rhs: &DimExpr) -> DimExpr;
}

#[declarations]
impl DimExpr {
    #[function(name = "lower", merge = |old: egg::I64, new: egg::I64| old.max(new))]
    pub fn lower(&self) -> egg::I64;

    #[function(name = "upper", merge = |old: egg::I64, new: egg::I64| old.min(new))]
    pub fn upper(&self) -> egg::I64;
}

#[declarations]
impl EList {
    #[function(name = "len", merge = |old: egg::I64, new: egg::I64| { let _ = old; new })]
    pub fn len(&self) -> egg::I64;

    #[function(name = "nth_from_end", merge = |old: DimExpr, new: DimExpr| { let _ = old; new })]
    pub fn nth_from_end(&self, index: egg::I64) -> DimExpr;

    #[function(name = "n_elements", merge = |old: DimExpr, new: DimExpr| { let _ = old; new })]
    pub fn n_elements(&self) -> DimExpr;
}

#[declarations]
impl Ir {
    #[function(name = "dtype", merge = |old: DType, new: DType| { let _ = old; new })]
    pub fn dtype(&self) -> DType;
}

#[declarations]
impl IList {
    #[relation(name = "identical_inputs")]
    pub fn identical_inputs(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_conversions_preserve_numeric_forms_and_payloads() {
        for value in [i64::MIN, -1, 0, i64::MAX] {
            assert_eq!(DimExpr::from(value), DimExpr::MNum(value));
            assert_eq!(DimExpr::from(&value), DimExpr::MNum(value));
        }
        for value in [i32::MIN, 0, i32::MAX] {
            assert_eq!(DimExpr::from(value), DimExpr::MNum(value));
            assert_eq!(DimExpr::from(&value), DimExpr::MNum(value));
        }
        let number = var::<egg::I64>("number");
        assert_eq!(DimExpr::from(&number), DimExpr::MNum(&number));
        assert_eq!(DimExpr::from(number.clone()), DimExpr::MNum(&number));
        assert_eq!(
            DimExpr::MNum(&number) + 2,
            DimExpr::MNum(&number) + DimExpr::MNum(2)
        );
        assert_eq!(
            DimExpr::MNum(2) * &number,
            DimExpr::MNum(2) * DimExpr::MNum(&number)
        );
        let x = DimExpr::from("x");
        assert_eq!(
            ((&x * &number) + 3) / &number,
            ((&x * DimExpr::MNum(&number)) + DimExpr::MNum(3)) / DimExpr::MNum(&number)
        );
        assert_eq!(
            &x - (&number - egg::I64::from(1)),
            &x - DimExpr::MNum(&number - egg::I64::from(1))
        );
        assert_eq!(
            DimExpr::MCeilDiv(&number, 0),
            DimExpr::MCeilDiv(DimExpr::MNum(&number), DimExpr::MNum(0))
        );
        // Conversion constructs syntax even when evaluating its input would fail.
        let quotient = egg::I64::from(1) / egg::I64::from(0);
        assert_eq!(DimExpr::from(&quotient), DimExpr::MNum(quotient));
        assert_eq!(&x / 0, &x / DimExpr::MNum(0));

        for bits in [0, 1 << 63, 0x7ff8_0000_0000_1234, f64::INFINITY.to_bits()] {
            let value = f64::from_bits(bits);
            let expression = DimExpr::from(value);
            assert_eq!(expression, DimExpr::MFloat(value));
            assert_eq!(DimExpr::from(&value), expression);
            let (field,) = get_args(&expression, |value: &egg::F64| DimExpr::MFloat(value))
                .unwrap()
                .unwrap();
            assert_eq!(f64::try_from(&field).unwrap().to_bits(), bits);
        }
        let float = var::<egg::F64>("float");
        assert_eq!(DimExpr::from(&float), DimExpr::MFloat(&float));
        assert_eq!(DimExpr::from(float.clone()), DimExpr::MFloat(float));
        assert_ne!(DimExpr::from(1), DimExpr::from(1.0));
    }

    #[test]
    fn name_conversions_author_data_instead_of_rule_or_iteration_variables() {
        let name = "quote\" slash\\ nul\0 λ😀".to_owned();
        let symbolic = egg::String::from(name.as_str());
        let expected = DimExpr::MVar(&symbolic);
        assert_eq!(DimExpr::from(name.as_str()), expected);
        assert_eq!(DimExpr::from(name.clone()), expected);
        assert_eq!(DimExpr::from(&name), expected);
        assert_eq!(DimExpr::from(&symbolic), expected);
        assert_eq!(DimExpr::from(symbolic), expected);
        let (field,) = get_args(&expected, |name: &egg::String| DimExpr::MVar(name))
            .unwrap()
            .unwrap();
        assert_eq!(String::try_from(&field).unwrap(), name);
        assert_ne!(DimExpr::from("x"), var::<DimExpr>("x"));
        for name in ['x', 'z', 'λ'] {
            assert_eq!(DimExpr::from(name), DimExpr::MVar(name.to_string()));
            assert_eq!(DimExpr::from(&name), DimExpr::from(name));
        }
        assert_ne!(DimExpr::from('z'), DimExpr::MIter());
    }

    #[test]
    fn host_dtypes_preserve_every_native_constructor() {
        use crate::dtype::DType as Host;
        for (host, expected) in [
            (Host::F32, DType::F32()),
            (Host::F64, DType::F64()),
            (Host::F16, DType::F16()),
            (Host::Bf16, DType::Bf16()),
            (Host::TF32, DType::TF32()),
            (Host::Int, DType::Int()),
            (Host::I64, DType::Int64()),
            (Host::I4, DType::I4()),
            (Host::U4, DType::U4()),
            (Host::I8, DType::I8()),
            (Host::U8, DType::U8()),
            (Host::I16, DType::I16()),
            (Host::U16, DType::U16()),
            (Host::Bool, DType::Bool()),
            (Host::F8UE8M0, DType::F8UE8M0()),
            (Host::F8E4M3, DType::F8E4M3()),
            (Host::F8E5M2, DType::F8E5M2()),
            (Host::F6E2M3, DType::F6E2M3()),
            (Host::F6E3M2, DType::F6E3M2()),
            (Host::F4E2M1, DType::F4E2M1()),
        ] {
            assert_eq!(DType::from(host), expected);
            assert_eq!(DType::from(&host), expected);
            assert_eq!(Ir::Input(0, "input", host), Ir::Input(0, "input", expected));
        }
    }

    #[test]
    fn ordered_lists_accept_owned_and_shallow_borrowed_collections() {
        let shared = DimExpr::from("x") + 1;
        let dimensions = [shared.clone(), DimExpr::from(2), shared.clone()];
        let expected = EList::ECons(
            &shared,
            EList::ECons(2, EList::ECons(&shared, EList::ENil())),
        );
        assert_eq!(EList::from(dimensions.clone()), expected);
        assert_eq!(EList::from(&dimensions), expected);
        assert_eq!(EList::from(dimensions.as_slice()), expected);
        let dimensions = dimensions.to_vec();
        assert_eq!(EList::from(&dimensions), expected);
        assert_eq!(EList::from(dimensions.clone()), expected);
        assert_eq!(dimensions.into_iter().collect::<EList>(), expected);
        assert_eq!(
            EList::from([1, 2]),
            EList::ECons(1, EList::ECons(2, EList::ENil()))
        );
        assert_eq!(
            EList::from(["x", "y"]),
            EList::ECons("x", EList::ECons("y", EList::ENil()))
        );
        assert_eq!(
            std::iter::empty::<DimExpr>().collect::<EList>(),
            EList::ENil()
        );
        assert_eq!(EList::from([] as [i64; 0]), EList::ENil());

        let first = Ir::Input(0, "first", crate::dtype::DType::F32);
        let second = Ir::Input(1, "second", crate::dtype::DType::I64);
        let sources = [&first, &second, &first];
        let expected = IList::ICons(
            &first,
            IList::ICons(&second, IList::ICons(&first, IList::INil())),
        );
        assert_eq!(IList::from(sources), expected);
        assert_eq!(IList::from(&sources), expected);
        assert_eq!(IList::from(sources.as_slice()), expected);
        let sources = sources.to_vec();
        assert_eq!(IList::from(&sources), expected);
        assert_eq!(IList::from(sources.clone()), expected);
        assert_eq!(sources.into_iter().collect::<IList>(), expected);
        assert_eq!(IList::from([] as [Ir; 0]), IList::INil());
    }

    #[test]
    fn list_elements_convert_once_in_source_order() {
        use std::cell::RefCell;
        struct Logged<'a>(i64, &'a RefCell<Vec<i64>>);
        impl From<Logged<'_>> for DimExpr {
            fn from(value: Logged<'_>) -> Self {
                value.1.borrow_mut().push(value.0);
                Self::MNum(value.0)
            }
        }
        let seen = RefCell::new(Vec::new());
        let array = [1, 2, 1].map(|value| Logged(value, &seen));
        assert_eq!(EList::from(array), EList::from([1, 2, 1]));
        assert_eq!(*seen.borrow(), [1, 2, 1]);
        seen.borrow_mut().clear();
        let list = [1, 2, 1]
            .into_iter()
            .map(|value| Logged(value, &seen))
            .collect::<EList>();
        assert_eq!(list, EList::from([1, 2, 1]));
        assert_eq!(*seen.borrow(), [1, 2, 1]);
    }

    #[test]
    fn conversions_keep_frozen_children_observation_only() -> Result<(), TypedError> {
        let scalar = let_("number", egg::I64::from(7));
        let input = let_("input", Ir::Input(0, "input", crate::dtype::DType::F32));
        let mut graph = EGraph::default();
        graph.register((&scalar, &input))?;
        let frozen = graph.freeze()?;
        let scalar = frozen.lookup(&scalar)?;
        let input = frozen.lookup(&input)?;
        let promoted = DimExpr::from(&scalar);
        let (field,) = get_args(&promoted, |value: &egg::I64| DimExpr::MNum(value))?.unwrap();
        assert_eq!(field, scalar);
        let references = [&input, &input];
        let list = IList::from(references.as_slice());
        let (head, tail) =
            get_args(&list, |head: &Ir, tail: &IList| IList::ICons(head, tail))?.unwrap();
        assert_eq!(head, input);
        assert_eq!(tail, IList::from([&input]));
        let before = graph.num_tuples()?;
        assert!(graph.register(promoted).is_err());
        assert!(graph.register(list).is_err());
        assert_eq!(graph.num_tuples()?, before);
        drop(frozen);
        drop(graph);
        assert_eq!(i64::try_from(&field).unwrap(), 7);
        assert_eq!(head, input);
        Ok(())
    }
}
