pub struct NetworkManager;

impl NetworkManager {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new()
    }
}
