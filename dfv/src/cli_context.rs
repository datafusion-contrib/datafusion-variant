// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use datafusion::{
    dataframe::DataFrame,
    error::Result,
    execution::{TaskContext, context::SessionState},
    logical_expr::LogicalPlan,
    object_store::ObjectStore,
    prelude::SessionContext,
};
use datafusion_cli::cli_context::CliSessionContext;

// TODO: remove this wrapper when CLI output uses the session's formatters:
// https://github.com/datafusion-contrib/datafusion-variant/issues/114
pub struct DisplayContext<'a>(pub &'a SessionContext);

#[async_trait::async_trait]
impl CliSessionContext for DisplayContext<'_> {
    fn task_ctx(&self) -> Arc<TaskContext> {
        self.0.task_ctx()
    }

    fn session_state(&self) -> SessionState {
        self.0.state()
    }

    fn register_object_store(
        &self,
        url: &url::Url,
        object_store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.0.register_object_store(url, object_store)
    }

    fn register_table_options_extension_from_scheme(&self, scheme: &str) {
        CliSessionContext::register_table_options_extension_from_scheme(self.0, scheme);
    }

    async fn execute_logical_plan(&self, plan: LogicalPlan) -> Result<DataFrame> {
        let df = self.0.execute_logical_plan(plan).await?;
        // Apply only to returned results, after DDL/DML execution. Query inputs,
        // stored tables, and COPY output keep their extension types.
        super::display::format_for_display(df, self.0.state().config_options().format.clone())
    }
}
