use foundations::telemetry::metrics::{metrics, Counter, Gauge, Histogram, HistogramBuilder, TimeHistogram};

#[metrics]
pub mod http_server {
    /// Number of HTTP requests
    pub fn requests_total(endpoint: &'static str) -> Counter;

    /// Number of HTTP requests that returned an error status
    pub fn requests_failed_total(endpoint: &'static str, status_code: u16) -> Counter;

    /// Distribution of response times per endpoint
    #[ctor = HistogramBuilder {
        buckets: &[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    }]
    pub fn response_time_seconds(endpoint: &'static str) -> TimeHistogram;
}

#[metrics]
pub mod recommend {
    /// Time spent running model inference, excluding queue wait
    #[ctor = HistogramBuilder {
        buckets: &[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    }]
    pub fn inference_time_seconds() -> TimeHistogram;

    /// Time spent waiting to acquire the inference lock
    #[ctor = HistogramBuilder {
        buckets: &[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
    }]
    pub fn queue_wait_seconds() -> TimeHistogram;

    /// Distribution of profile sizes (raw entry counts) in recommend requests
    #[ctor = HistogramBuilder {
        buckets: &[5., 10., 25., 50., 100., 175., 250., 375., 500., 750., 1000., 1500., 2500.],
    }]
    pub fn profile_size() -> Histogram;

    /// Distribution of serialized response sizes
    #[ctor = HistogramBuilder {
        buckets: &[1024., 4096., 16384., 65536., 262144., 1048576., 4194304., 16777216.],
    }]
    pub fn response_size_bytes() -> Histogram;

    /// Requests using optional request features
    pub fn feature_requests_total(feature: &'static str) -> Counter;

    /// Requests rejected due to empty or invalid profiles
    pub fn invalid_profile_total() -> Counter;

    /// Inference tasks that panicked
    pub fn inference_panics_total() -> Counter;
}

#[metrics]
pub mod cache {
    /// Number of response cache hits
    pub fn hits_total() -> Counter;

    /// Number of response cache misses
    pub fn misses_total() -> Counter;

    /// Number of response cache evictions
    pub fn evictions_total() -> Counter;

    /// Number of entries currently in the response cache
    pub fn entries() -> Gauge;

    /// Approximate bytes held by the response cache (keys + serialized responses)
    pub fn size_bytes() -> Gauge;
}
