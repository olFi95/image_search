use crate::server_arguments::ServerArguments;
use surrealdb::engine::any::Any;
use surrealdb::opt::auth::Root;
use surrealdb::{Error, Surreal};
pub async fn init_database(
    cla: &ServerArguments,
) -> Result<Surreal<Any>, Error> {
    let db: Surreal<Any> = Surreal::init();
    db.connect(cla.surrealdb_uri.clone()).await?;


    if !(cla.surrealdb_uri.starts_with("mem://")
        || cla.surrealdb_uri.starts_with("rocksdb://")) {
        db.signin(Root {
                username: cla.surrealdb_username.clone(),
                password: cla.surrealdb_password.clone(),
            })
                .await?;

    };
    db.use_ns(&cla.surrealdb_namespace)
        .use_db(&cla.surrealdb_database)
        .await?;

    Ok(db)
}