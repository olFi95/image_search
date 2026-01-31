use zbus::proxy;
use std::collections::HashMap;
use zbus::{Connection};
use zvariant::Value;

#[proxy(
    default_service = "org.freedesktop.Notifications",
    default_path = "/org/freedesktop/Notifications"
)]
trait Notifications {
    fn notify(&self,
              app_name: &str,
              replaces_id: u32,
              app_icon: &str,
              summary: &str,
              body: &str,
              actions: &[&str],
              hints: HashMap<&str, &zvariant::Value<'_>>,
              expire_timeout: i32) -> zbus::Result<u32>;
}

pub struct Toaster;

impl Toaster {
    pub async fn toast(event_id: Option<u32>, summary: &str, body: &str, urgent: bool) -> zbus::Result<u32> {
        let conn = Connection::session().await?;

        let proxy = NotificationsProxy::new(&conn).await?;

        let mut hints = HashMap::<&str, &Value>::new();
        let binding = match urgent {
            true => Value::from(2u8),
            false => Value::from(1u8)
        };
        hints.insert("urgency", &binding); // critical

        let id = proxy
            .notify(
                "image-search",
                event_id.unwrap_or(0),
                "dialog-information",
                summary,
                body,
                &[],
                hints,
                5000,
            )
            .await?;

        Ok(id)
    }
}