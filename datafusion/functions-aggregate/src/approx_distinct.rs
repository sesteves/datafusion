// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines physical expressions that can evaluated at runtime during query execution

use crate::hyperloglog::{HLL_HASH_STATE, HyperLogLog};
use arrow::array::{Array, FixedSizeBinaryArray, StringViewArray};
use arrow::array::{
    GenericBinaryArray, GenericStringArray, OffsetSizeTrait, PrimitiveArray,
};
use arrow::datatypes::{
    ArrowPrimitiveType, Date32Type, Date64Type, FieldRef, Int32Type, Int64Type,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
    TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt32Type, UInt64Type,
};
use arrow::{array::ArrayRef, datatypes::DataType, datatypes::Field};
use datafusion_common::ScalarValue;
use datafusion_common::{
    DataFusionError, Result, downcast_value, internal_datafusion_err, internal_err,
    not_impl_err,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::count_distinct::{
    Bitmap65536DistinctCountAccumulator, Bitmap65536DistinctCountAccumulatorI16,
    BoolArray256DistinctCountAccumulator, BoolArray256DistinctCountAccumulatorI8,
};
use datafusion_functions_aggregate_common::noop_accumulator::NoopAccumulator;
use datafusion_macros::user_doc;
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

/// Number of registers in the dense HyperLogLog intermediate state.
pub const APPROX_DISTINCT_HLL_STATE_SIZE: i32 = 16384;

make_udaf_expr_and_func!(
    ApproxDistinct,
    approx_distinct,
    expression,
    "approximate number of distinct input values",
    approx_distinct_udaf
);

impl<T: Hash + ?Sized> From<&HyperLogLog<T>> for ScalarValue {
    fn from(v: &HyperLogLog<T>) -> ScalarValue {
        let values = v.as_ref().to_vec();
        ScalarValue::FixedSizeBinary(APPROX_DISTINCT_HLL_STATE_SIZE, Some(values))
    }
}

impl<T: Hash + ?Sized> TryFrom<&[u8]> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &[u8]) -> Result<HyperLogLog<T>> {
        let arr: [u8; APPROX_DISTINCT_HLL_STATE_SIZE as usize] = v.try_into().map_err(|_| {
            internal_datafusion_err!(
                "approx_distinct HLL state has length {}, expected {APPROX_DISTINCT_HLL_STATE_SIZE}",
                v.len()
            )
        })?;
        Ok(HyperLogLog::<T>::new_with_registers(arr))
    }
}

impl<T: Hash + ?Sized> TryFrom<&ScalarValue> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &ScalarValue) -> Result<HyperLogLog<T>> {
        match v {
            ScalarValue::FixedSizeBinary(width, Some(value))
                if *width == APPROX_DISTINCT_HLL_STATE_SIZE =>
            {
                value.as_slice().try_into()
            }
            ScalarValue::FixedSizeBinary(width, _) => internal_err!(
                "approx_distinct HLL state has width {width}, expected {APPROX_DISTINCT_HLL_STATE_SIZE}"
            ),
            _ => internal_err!(
                "approx_distinct HLL state must be FixedSizeBinary({APPROX_DISTINCT_HLL_STATE_SIZE})"
            ),
        }
    }
}

#[derive(Debug)]
struct ApproxDistinctBitmapWrapper<A: Accumulator> {
    inner: A,
}

impl<A: Accumulator> Accumulator for ApproxDistinctBitmapWrapper<A> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.update_batch(values)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        match self.inner.evaluate()? {
            ScalarValue::Int64(Some(v)) => Ok(ScalarValue::UInt64(Some(v as u64))),
            other => internal_err!("unexpected: {other}"),
        }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.merge_batch(states)
    }
}

#[derive(Debug)]
struct NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    hll: HyperLogLog<T::Native>,
}

impl<T> NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<str>,
    phantom_data: PhantomData<T>,
}

impl<T> StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

#[derive(Debug)]
struct StringViewHLLAccumulator {
    hll: HyperLogLog<str>,
}

impl StringViewHLLAccumulator {
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<[u8]>,
    phantom_data: PhantomData<T>,
}

impl<T> BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

macro_rules! default_accumulator_impl {
    () => {
        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            if states.len() != 1 {
                return internal_err!(
                    "approx_distinct expects one HLL state array, got {}",
                    states.len()
                );
            }
            let binary_array = states[0]
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "approx_distinct HLL state must be FixedSizeBinary({APPROX_DISTINCT_HLL_STATE_SIZE}), got {}",
                        states[0].data_type()
                    )
                })?;
            if binary_array.value_length() != APPROX_DISTINCT_HLL_STATE_SIZE {
                return internal_err!(
                    "approx_distinct HLL state has width {}, expected {APPROX_DISTINCT_HLL_STATE_SIZE}",
                    binary_array.value_length()
                );
            }
            for v in binary_array.iter() {
                let v = v.ok_or_else(|| {
                    internal_datafusion_err!(
                        "approx_distinct HLL state must not be null"
                    )
                })?;
                let other = v.try_into()?;
                self.hll.merge(&other);
            }
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<ScalarValue>> {
            let value = ScalarValue::from(&self.hll);
            Ok(vec![value])
        }

        fn evaluate(&mut self) -> Result<ScalarValue> {
            Ok(ScalarValue::UInt64(Some(self.hll.count() as u64)))
        }

        fn size(&self) -> usize {
            // HLL has static size
            std::mem::size_of_val(self)
        }
    };
}

impl<T> Accumulator for BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericBinaryArray<T> =
            downcast_value!(values[0], GenericBinaryArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Accumulator for StringViewHLLAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &StringViewArray = downcast_value!(values[0], StringViewArray);

        if array.data_buffers().is_empty() {
            // Fast path: with no data buffers every value is inline, so they all
            // take the u128 path — no need to check the length per row.
            for (i, &view) in array.views().iter().enumerate() {
                if !array.is_null(i) {
                    self.hll.add_hashed(HLL_HASH_STATE.hash_one(view));
                }
            }
        } else {
            // Mixed batch: decide per row by length. Short strings still use the
            // u128 path so they match how they'd be hashed in an all-inline
            // batch; only the genuinely out-of-line strings materialize a &str.
            for (i, &view) in array.views().iter().enumerate() {
                if array.is_null(i) {
                    continue;
                }
                // The low 32 bits of the u128 view encode the string length.
                if (view as u32) <= 12 {
                    self.hll.add_hashed(HLL_HASH_STATE.hash_one(view));
                } else {
                    self.hll.add(array.value(i));
                }
            }
        }

        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericStringArray<T> =
            downcast_value!(values[0], GenericStringArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType + Debug,
    T::Native: Hash,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &PrimitiveArray<T> = downcast_value!(values[0], PrimitiveArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Debug for ApproxDistinct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxDistinct")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .finish()
    }
}

impl Default for ApproxDistinct {
    fn default() -> Self {
        Self::new()
    }
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate number of distinct input values calculated using the HyperLogLog algorithm.",
    syntax_example = "approx_distinct(expression)",
    sql_example = r#"```sql
> SELECT approx_distinct(column_name) FROM table_name;
+-----------------------------------+
| approx_distinct(column_name)      |
+-----------------------------------+
| 42                                |
+-----------------------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(PartialEq, Eq, Hash)]
pub struct ApproxDistinct {
    signature: Signature,
}

impl ApproxDistinct {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

#[cold]
fn get_small_int_approx_accumulator(
    data_type: &DataType,
) -> Result<Box<dyn Accumulator>> {
    match data_type {
        DataType::UInt8 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: BoolArray256DistinctCountAccumulator::new(),
        })),
        DataType::Int8 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: BoolArray256DistinctCountAccumulatorI8::new(),
        })),
        DataType::UInt16 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: Bitmap65536DistinctCountAccumulator::new(),
        })),
        DataType::Int16 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: Bitmap65536DistinctCountAccumulatorI16::new(),
        })),
        _ => internal_err!("unsupported small int type: {}", data_type),
    }
}

#[cold]
fn get_small_int_state_field(name: &str, data_type: &DataType) -> Result<Vec<FieldRef>> {
    Ok(vec![
        Field::new_list(
            format_state_name(name, "approx_distinct"),
            Field::new_list_field(data_type.clone(), true),
            false,
        )
        .into(),
    ])
}

impl AggregateUDFImpl for ApproxDistinct {
    fn name(&self) -> &str {
        "approx_distinct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let data_type = args.input_fields[0].data_type();
        match data_type {
            DataType::Null => Ok(vec![
                Field::new(
                    format_state_name(args.name, self.name()),
                    DataType::Null,
                    true,
                )
                .into(),
            ]),
            DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16 => {
                get_small_int_state_field(args.name, data_type)
            }
            _ => Ok(vec![
                Field::new(
                    format_state_name(args.name, "hll_registers"),
                    DataType::FixedSizeBinary(APPROX_DISTINCT_HLL_STATE_SIZE),
                    false,
                )
                .into(),
            ]),
        }
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let data_type = acc_args.expr_fields[0].data_type();

        let accumulator: Box<dyn Accumulator> = match data_type {
            DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16 => {
                return get_small_int_approx_accumulator(data_type);
            }
            DataType::UInt32 => Box::new(NumericHLLAccumulator::<UInt32Type>::new()),
            DataType::UInt64 => Box::new(NumericHLLAccumulator::<UInt64Type>::new()),
            DataType::Int32 => Box::new(NumericHLLAccumulator::<Int32Type>::new()),
            DataType::Int64 => Box::new(NumericHLLAccumulator::<Int64Type>::new()),
            DataType::Date32 => Box::new(NumericHLLAccumulator::<Date32Type>::new()),
            DataType::Date64 => Box::new(NumericHLLAccumulator::<Date64Type>::new()),
            DataType::Time32(TimeUnit::Second) => {
                Box::new(NumericHLLAccumulator::<Time32SecondType>::new())
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                Box::new(NumericHLLAccumulator::<Time32MillisecondType>::new())
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                Box::new(NumericHLLAccumulator::<Time64MicrosecondType>::new())
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                Box::new(NumericHLLAccumulator::<Time64NanosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                Box::new(NumericHLLAccumulator::<TimestampSecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMillisecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMicrosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampNanosecondType>::new())
            }
            DataType::Utf8 => Box::new(StringHLLAccumulator::<i32>::new()),
            DataType::LargeUtf8 => Box::new(StringHLLAccumulator::<i64>::new()),
            DataType::Utf8View => Box::new(StringViewHLLAccumulator::new()),
            DataType::Binary => Box::new(BinaryHLLAccumulator::<i32>::new()),
            DataType::LargeBinary => Box::new(BinaryHLLAccumulator::<i64>::new()),
            DataType::Null => {
                Box::new(NoopAccumulator::new(ScalarValue::UInt64(Some(0))))
            }
            other => {
                return not_impl_err!(
                    "Support for 'approx_distinct' for data type {other} is not implemented"
                );
            }
        };
        Ok(accumulator)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Int64Array};
    use std::sync::Arc;

    // A string longer than the 12-byte inline limit
    const LONG: &str = "this string is definitely longer than twelve bytes";

    fn distinct_count(acc: &mut StringViewHLLAccumulator) -> u64 {
        match acc.evaluate().unwrap() {
            ScalarValue::UInt64(Some(v)) => v,
            other => panic!("unexpected evaluate result: {other:?}"),
        }
    }

    fn state_fields(input_type: DataType) -> Vec<FieldRef> {
        let input_fields = vec![Arc::new(Field::new("input", input_type, true))];
        ApproxDistinct::new()
            .state_fields(StateFieldsArgs {
                name: "approx_distinct",
                input_fields: &input_fields,
                return_field: Arc::new(Field::new("result", DataType::UInt64, false)),
                ordering_fields: &[],
                is_distinct: false,
            })
            .unwrap()
    }

    #[test]
    fn hll_state_schema_is_fixed_size_binary() {
        let fields = state_fields(DataType::Int64);
        assert_eq!(
            fields[0].data_type(),
            &DataType::FixedSizeBinary(APPROX_DISTINCT_HLL_STATE_SIZE)
        );
        assert!(!fields[0].is_nullable());

        let fields = state_fields(DataType::Int16);
        assert!(matches!(fields[0].data_type(), DataType::List(_)));
    }

    #[test]
    fn hll_state_round_trip() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 2, 3]));
        let mut source = NumericHLLAccumulator::<Int64Type>::new();
        source.update_batch(&[values]).unwrap();
        let expected = source.evaluate().unwrap();
        let state = source.state().unwrap();
        assert!(matches!(
            &state[0],
            ScalarValue::FixedSizeBinary(APPROX_DISTINCT_HLL_STATE_SIZE, Some(value))
                if value.len() == APPROX_DISTINCT_HLL_STATE_SIZE as usize
        ));

        let state_array = ScalarValue::iter_to_array(state).unwrap();
        let mut merged = NumericHLLAccumulator::<Int64Type>::new();
        merged.merge_batch(&[state_array]).unwrap();
        assert_eq!(merged.evaluate().unwrap(), expected);
    }

    #[test]
    fn malformed_hll_state_is_rejected() {
        let malformed = ScalarValue::FixedSizeBinary(
            APPROX_DISTINCT_HLL_STATE_SIZE,
            Some(vec![0; APPROX_DISTINCT_HLL_STATE_SIZE as usize - 1]),
        );
        let error = HyperLogLog::<i64>::try_from(&malformed).unwrap_err();
        let malformed_length = APPROX_DISTINCT_HLL_STATE_SIZE - 1;
        assert!(error.to_string().contains(&format!(
            "has length {malformed_length}, expected {APPROX_DISTINCT_HLL_STATE_SIZE}"
        )));

        let wrong_width = ScalarValue::FixedSizeBinary(8, Some(vec![0; 8]));
        let error = HyperLogLog::<i64>::try_from(&wrong_width).unwrap_err();
        let wrong_width_message =
            format!("has width 8, expected {APPROX_DISTINCT_HLL_STATE_SIZE}");
        assert!(error.to_string().contains(&wrong_width_message));

        let wrong_width_array = ScalarValue::iter_to_array(vec![wrong_width]).unwrap();
        let mut acc = NumericHLLAccumulator::<Int64Type>::new();
        let error = acc.merge_batch(&[wrong_width_array]).unwrap_err();
        assert!(error.to_string().contains(&wrong_width_message));

        let null_state = ScalarValue::iter_to_array(vec![ScalarValue::FixedSizeBinary(
            APPROX_DISTINCT_HLL_STATE_SIZE,
            None,
        )])
        .unwrap();
        let error = acc.merge_batch(&[null_state]).unwrap_err();
        assert!(error.to_string().contains("must not be null"));
    }

    /// Regression: a short (≤ 12-byte) Utf8View string must hash identically
    /// regardless of which batch it appears in — all-inline or mixed.
    #[test]
    fn utf8view_acc_split_batches_match_single_mixed_batch() {
        // Multiset: {"aaa" x2, "bbb", LONG}, so 3 distinct values.
        let mixed: ArrayRef =
            Arc::new(StringViewArray::from(vec!["aaa", "bbb", LONG, "aaa"]));
        let mut acc_single = StringViewHLLAccumulator::new();
        acc_single.update_batch(&[mixed]).unwrap();

        // Same multiset, but split so "aaa" lands in both an all-inline batch
        // and a batch with a data buffer (forced by LONG).
        let inline_only: ArrayRef = Arc::new(StringViewArray::from(vec!["aaa", "bbb"]));
        let with_buffer: ArrayRef = Arc::new(StringViewArray::from(vec!["aaa", LONG]));
        assert!(inline_only.as_string_view().data_buffers().is_empty());
        assert!(!with_buffer.as_string_view().data_buffers().is_empty());

        let mut acc_split = StringViewHLLAccumulator::new();
        acc_split.update_batch(&[inline_only]).unwrap();
        acc_split.update_batch(&[with_buffer]).unwrap();

        assert_eq!(
            distinct_count(&mut acc_single),
            distinct_count(&mut acc_split)
        );
        assert_eq!(distinct_count(&mut acc_single), 3);
    }
}
