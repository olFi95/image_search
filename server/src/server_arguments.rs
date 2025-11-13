use std::net::{IpAddr, SocketAddr};
use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct ServerArguments {
    #[clap(short = 'm', long = "media-dir", default_value = "~/Pictures")]
    pub media_dir: String,
    #[clap(short = 'a', long = "addr", default_value = "127.0.0.1")]
    pub addr: String,
    #[clap(short = 'p', long = "port", default_value_t = 3000)]
    pub port: u16,
    #[clap(long = "surrealdb-uri", default_value = "localhost:8000")]
    pub surrealdb_uri: String,
    #[clap(long = "surrealdb-username", default_value = "root")]
    pub surrealdb_username: String,
    #[clap(long = "surrealdb-password", default_value = "root")]
    pub surrealdb_password: String,
    #[clap(long = "surrealdb-namespace", default_value = "pictures")]
    pub surrealdb_namespace: String,
    #[clap(long = "surrealdb-database", default_value = "pictures")]
    pub surrealdb_database: String,
}

impl ServerArguments {
    pub fn get_socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.addr.parse().unwrap(), self.port)
    }
}