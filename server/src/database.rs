use crate::server_arguments::ServerArguments;
use log::info;
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::opt::auth::Root;
use surrealdb::{Error, Surreal};

pub async fn init_database(cla: &ServerArguments) -> Result<Surreal<Client>, Error> {
    let surrealdb = Surreal::new::<Ws>(cla.surrealdb_uri.clone()).await?;
    surrealdb
        .signin(Root {
            username: &cla.surrealdb_username,
            password: &cla.surrealdb_password,
        })
        .await?;
    surrealdb
        .use_ns(&cla.surrealdb_namespace)
        .use_db(&cla.surrealdb_database)
        .await
        .unwrap();
    info!("SurrealDB initialized");
    Ok(surrealdb)
}
