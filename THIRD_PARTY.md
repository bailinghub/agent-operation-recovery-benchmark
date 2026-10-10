# Third-party dependencies

No third-party SDK implementation or model weights are included in this package. Dependencies are installed separately and retain their own licenses and copyright notices.

The benchmark uses the public interfaces of:

- [Kaggle Benchmarks](https://github.com/Kaggle/kaggle-benchmarks) for tasks, native tools, model actors, and evidence.
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and [Kaggle SDK](https://github.com/Kaggle/kagglesdk) as dependencies of the recorded environment.
- [OpenAI Python SDK](https://github.com/openai/openai-python) as an OpenAI-compatible HTTP client for the external provider.

The reference lock also lists transitive dependencies such as HTTPX and Pydantic. Their inclusion in the environment snapshot is not redistribution of their implementation. Inspect each installed distribution's license metadata and upstream license before redistributing dependency files.

The external provider was Tencent TokenHub. Using its endpoint requires your own account and is governed by its service terms. Neither the benchmark code license nor the result archive grants access to that service.
