use std::fs::File;

use datafusion::prelude::SessionContext;
use flate2::read::GzDecoder;
use std::io::{BufRead, BufReader};

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::ScalarUDF;
use datafusion_variant::{JsonToVariantUdf, VariantGetUdf};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut args = std::env::args();

    let num_rows = {
        args.next();
        let n_str = args.next().expect("expected argument specifiying num rows");

        n_str.parse::<usize>().expect("expected number")
    };

    let file_path = "data/bluesky/file_0001.json.gz";

    // load data
    let file = File::open(file_path).expect("make sure to run ./bin/download_data.sh");
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let json_strings = reader
        .lines()
        .take(num_rows)
        .map(|l| l.unwrap())
        .collect::<Vec<_>>();

    let ctx = SessionContext::new();
    let schema = Schema::new(vec![Field::new("json_data", DataType::Utf8, false)]);
    let string_array: ArrayRef = Arc::new(StringArray::from(json_strings));
    let batch = RecordBatch::try_new(Arc::new(schema), vec![string_array]).unwrap();

    let provider =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();

    ctx.register_table("bsky", Arc::new(provider)).unwrap();

    // register variant udfs
    ctx.register_udf(ScalarUDF::new_from_impl(JsonToVariantUdf::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));

    let _df = ctx
        .sql(
            r"
        select json_to_variant(bsky.json_data) from bsky
    ",
        )
        .await
        .unwrap();
}
