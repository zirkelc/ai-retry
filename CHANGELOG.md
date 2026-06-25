# Changelog

## [2.0.0](https://github.com/zirkelc/ai-retry/compare/v1.11.1...v2.0.0) (2026-06-25)


### ⚠ BREAKING CHANGES

* requires AI SDK v7 (ai@7, @ai-sdk/provider@4, @ai-sdk/provider-utils@5). Release-As: 2.0.0-beta.0

### Features

* AI SDK v7 support ([#68](https://github.com/zirkelc/ai-retry/issues/68)) ([95feb92](https://github.com/zirkelc/ai-retry/commit/95feb92d20583b7a9089b75cae9c70f6dd69f5f0))

## [1.11.1](https://github.com/zirkelc/ai-retry/compare/v1.11.0...v1.11.1) (2026-06-25)


### Documentation

* document AI SDK v7 beta release and install tag ([e5c0ebd](https://github.com/zirkelc/ai-retry/commit/e5c0ebd28163779384327061bc6efa912d0e0a96))


### Tests

* adopt ai-test-kit and refactor test fixtures ([a11c3fb](https://github.com/zirkelc/ai-retry/commit/a11c3fbbee0c5fee46388e64c2365f70e9431f41))
* verify nested and/or/not condition composition ([d7ebee6](https://github.com/zirkelc/ai-retry/commit/d7ebee6ad7d57166cc50a59ba32d7695685d2ef1))


### Miscellaneous Chores

* upgrade ai-test-kit to 2.0.0 ([9f69749](https://github.com/zirkelc/ai-retry/commit/9f6974960051f6418ec3eb73c2890c4312e69e6d))

## [1.11.0](https://github.com/zirkelc/ai-retry/compare/v1.10.0...v1.11.0) (2026-06-17)


### Features

* promote condition-based retryables to the primary API ([298f050](https://github.com/zirkelc/ai-retry/commit/298f050955f842902dca1b8d99ad624bf65fa7e8))

## [1.10.0](https://github.com/zirkelc/ai-retry/compare/v1.9.1...v1.10.0) (2026-06-08)


### Features

* add onFailure callback as counterpart to onSuccess ([#56](https://github.com/zirkelc/ai-retry/issues/56)) ([ffec8d2](https://github.com/zirkelc/ai-retry/commit/ffec8d25261572b17bd6950d618b7fde7c49f572))

## [1.9.1](https://github.com/zirkelc/ai-retry/compare/v1.9.0...v1.9.1) (2026-06-02)


### Bug Fixes

* **stream:** recover mid-stream retry before first content ([956f926](https://github.com/zirkelc/ai-retry/commit/956f92627d5142e6e3d8c08247ba95ef29ed2ca2))


### Documentation

* **readme:** explain stream preamble buffering ([0018a4b](https://github.com/zirkelc/ai-retry/commit/0018a4be66966139129c5391af4f50eee8245059))

## [1.9.0](https://github.com/zirkelc/ai-retry/compare/v1.8.0...v1.9.0) (2026-05-27)


### Features

* **telemetry:** record HTTP status on error spans ([032c6d1](https://github.com/zirkelc/ai-retry/commit/032c6d15de84dfdb80d33211ed65fe343d2eb7ec))

## [1.8.0](https://github.com/zirkelc/ai-retry/compare/v1.7.4...v1.8.0) (2026-05-25)


### Features

* **telemetry:** add OpenTelemetry instrumentation for retries ([ed4647f](https://github.com/zirkelc/ai-retry/commit/ed4647f5ad63598300e9c1e2d6a067e3c9140e52))

## [1.7.4](https://github.com/zirkelc/ai-retry/compare/v1.7.3...v1.7.4) (2026-05-18)


### Bug Fixes

* don't let parent AbortSignal.timeout truncate retry deadlines ([f5ae11c](https://github.com/zirkelc/ai-retry/commit/f5ae11c595cea9c76bdabad09c18a8a769577f43)), closes [#46](https://github.com/zirkelc/ai-retry/issues/46)


### Documentation

* **experimental:** add terminal-action note to every condition helper ([567713c](https://github.com/zirkelc/ai-retry/commit/567713c3c1bd433109f72a749effac184d809082))

## [1.7.3](https://github.com/zirkelc/ai-retry/compare/v1.7.2...v1.7.3) (2026-05-11)


### Bug Fixes

* **experimental:** accept gateway-string fallbacks in conditions ([59c89b1](https://github.com/zirkelc/ai-retry/commit/59c89b15e3acbbe3598ad13cc6095d6c82e6695e))

## [1.7.2](https://github.com/zirkelc/ai-retry/compare/v1.7.1...v1.7.2) (2026-05-11)


### Code Refactoring

* **experimental:** bound result API factory mirroring error ([840913c](https://github.com/zirkelc/ai-retry/commit/840913cc27b9f79e8e4ecf7e33fdff59db3453ca))


### Tests

* **streamText:** cover content-filter finish after streamed content ([f19ce54](https://github.com/zirkelc/ai-retry/commit/f19ce547943d6e5ca926cacc70827ed5bf6d689f))


### Continuous Integration

* **release-please:** switch to manifest config, drop v6 branch ([8ba1a34](https://github.com/zirkelc/ai-retry/commit/8ba1a347d3eefa7e9a595be4212057b959b7415e))

## [1.7.1](https://github.com/zirkelc/ai-retry/compare/v1.7.0...v1.7.1) (2026-05-08)


### Bug Fixes

* surface mid-stream errors as stream parts when no retryable matches ([3d61efd](https://github.com/zirkelc/ai-retry/commit/3d61efd4b0321c3a5d1c28c4c00a1904694ec56e)), closes [#42](https://github.com/zirkelc/ai-retry/issues/42)

## [1.7.0](https://github.com/zirkelc/ai-retry/compare/v1.6.1...v1.7.0) (2026-05-07)


### Features

* **experimental:** add result.finishReason and polish docs ([35f33e8](https://github.com/zirkelc/ai-retry/commit/35f33e86d462ff57c2388f50c364500280eee8c7))
* **experimental:** split conditions into per-model entry points ([7e53903](https://github.com/zirkelc/ai-retry/commit/7e53903d69c4ad4dc83a6f0ffd883c6a0e5b23d1))
* result-based retries for streamText ([7c0270b](https://github.com/zirkelc/ai-retry/commit/7c0270bcd82acfdaf7cd39e67e577eadfbee295e))


### Bug Fixes

* tests ([4bc738f](https://github.com/zirkelc/ai-retry/commit/4bc738f5296e505b6b8ef73591e7fa2484b20aed))

## [1.6.1](https://github.com/zirkelc/ai-retry/compare/v1.6.0...v1.6.1) (2026-05-04)


### Bug Fixes

* skip retry when inbound signal is aborted and retry has no timeout ([814de8e](https://github.com/zirkelc/ai-retry/commit/814de8ed151d0f069d2446a51083818fd87554af))

## [1.6.0](https://github.com/zirkelc/ai-retry/compare/v1.5.0...v1.6.0) (2026-04-28)


### Features

* add experimental composable conditions retryable API ([370bb03](https://github.com/zirkelc/ai-retry/commit/370bb033390abc5907a4f97737216a3690b2cbec))


### Bug Fixes

* ci ([869d918](https://github.com/zirkelc/ai-retry/commit/869d918bfb92fd866a8c7bf9f9b9a0d646d7db35))

## [1.5.0](https://github.com/zirkelc/ai-retry/compare/v1.4.0...v1.5.0) (2026-04-14)


### Features

* allow onRetry to return overrides for the upcoming retry attempt ([9de8f7d](https://github.com/zirkelc/ai-retry/commit/9de8f7d2d7b218bb99bb8d3057af28c779c7759a)), closes [#36](https://github.com/zirkelc/ai-retry/issues/36)


### Bug Fixes

* lint ([6ed96e3](https://github.com/zirkelc/ai-retry/commit/6ed96e3c0c600d3d8e11f01e00eb4520dee6a971))
* lint ci ([73a61f8](https://github.com/zirkelc/ai-retry/commit/73a61f898d22f4a1b6b19d2e84643426f8231970))

## [1.4.0](https://github.com/zirkelc/ai-retry/compare/v1.3.1...v1.4.0) (2026-04-01)


### Features

* add onSuccess callback to expose which model handled the request ([02b0766](https://github.com/zirkelc/ai-retry/commit/02b076690bd372276c09205128826e6fcdff772b)), closes [#31](https://github.com/zirkelc/ai-retry/issues/31)

## [1.3.1](https://github.com/zirkelc/ai-retry/compare/v1.3.0...v1.3.1) (2026-04-01)


### Bug Fixes

* move @ai-sdk/provider to peerDependencies to avoid type conflicts ([2046e09](https://github.com/zirkelc/ai-retry/commit/2046e090fdcbc37b70fd486c04f0af0ef59fd563)), closes [#32](https://github.com/zirkelc/ai-retry/issues/32)

## [1.3.0](https://github.com/zirkelc/ai-retry/compare/v1.2.0...v1.3.0) (2026-02-27)


### Features

* add reryable image model ([9e93d26](https://github.com/zirkelc/ai-retry/commit/9e93d2687b0fd7df4286f9564c602431c23e1473))

## [1.2.0](https://github.com/zirkelc/ai-retry/compare/v1.1.0...v1.2.0) (2026-01-28)


### Features

* add reset option for managing sticky models ([5725c79](https://github.com/zirkelc/ai-retry/commit/5725c79967930566ad623eb41d0bff60b7ecfdca))

## [1.1.0](https://github.com/zirkelc/ai-retry/compare/v1.0.2...v1.1.0) (2026-01-27)


### Features

* add schemaMismatch retryable ([7bcb7a7](https://github.com/zirkelc/ai-retry/commit/7bcb7a718ed061322b0b7c5b34e969d62dde6895))


### Bug Fixes

* add Zod ([aec5f69](https://github.com/zirkelc/ai-retry/commit/aec5f693bf572fa260af2f09183ce2c767f1fb57))
* tests ([d1cda2f](https://github.com/zirkelc/ai-retry/commit/d1cda2f3f9cef1d96fb21c6d6e321511de545c57))

## [1.0.2](https://github.com/zirkelc/ai-retry/compare/v1.0.1...v1.0.2) (2026-01-12)


### Bug Fixes

* do not retry user aborted requests ([02cf1e5](https://github.com/zirkelc/ai-retry/commit/02cf1e558a4cda9a3778110009bcb4a710fd08e9)), closes [#21](https://github.com/zirkelc/ai-retry/issues/21)

## [1.0.1](https://github.com/zirkelc/ai-retry/compare/v1.0.0...v1.0.1) (2025-12-23)


### Bug Fixes

* update readme ([23951c8](https://github.com/zirkelc/ai-retry/commit/23951c8d40ed5210618f81e5bd92ae11b4ba14e7))

## [1.0.0](https://github.com/zirkelc/ai-retry/compare/v0.12.0...v1.0.0) (2025-12-22)


### ⚠ BREAKING CHANGES

* AI SDK v6 upgrade

### Features

* AI SDK v6 upgrade ([01b9c62](https://github.com/zirkelc/ai-retry/commit/01b9c6247439459d44068b1124c2239dd8ebd66d))


### Bug Fixes

* esm only ([8543a89](https://github.com/zirkelc/ai-retry/commit/8543a89b31807b57ef467b773c645f5e1a71fe02))

## [0.12.0](https://github.com/zirkelc/ai-retry/compare/v0.11.0...v0.12.0) (2025-12-10)


### Features

* support ai gateway model strings ([a6ad6a9](https://github.com/zirkelc/ai-retry/commit/a6ad6a9dabfc6ad1ba325a2782a8b12c35924af9))

## [0.11.0](https://github.com/zirkelc/ai-retry/compare/v0.10.3...v0.11.0) (2025-11-30)


### Features

* override call options ([7074aee](https://github.com/zirkelc/ai-retry/commit/7074aee14c3e738bbc6e0461436447bd80016589))

## [0.10.3](https://github.com/zirkelc/ai-retry/compare/v0.10.2...v0.10.3) (2025-11-26)


### Bug Fixes

* support promise&lt;undefined&gt; ([9673c19](https://github.com/zirkelc/ai-retry/commit/9673c19866980b66d94ad378e8d08b963e54cf06))

## [0.10.2](https://github.com/zirkelc/ai-retry/compare/v0.10.1...v0.10.2) (2025-11-26)


### Bug Fixes

* refactor types ([8ecadc4](https://github.com/zirkelc/ai-retry/commit/8ecadc4f3b39da6388080ee0365ef736bc6e5da6))

## [0.10.1](https://github.com/zirkelc/ai-retry/compare/v0.10.0...v0.10.1) (2025-11-24)


### Bug Fixes

* missing types ([24bf204](https://github.com/zirkelc/ai-retry/commit/24bf204489ee4e0d6d1db324ccca4bf6ab7a385f))

## [0.10.0](https://github.com/zirkelc/ai-retry/compare/v0.9.0...v0.10.0) (2025-11-24)


### Features

* disable retries ([5386801](https://github.com/zirkelc/ai-retry/commit/5386801b4545eb7439b9025546bfee4f63c1476e))

## [0.9.0](https://github.com/zirkelc/ai-retry/compare/v0.8.0...v0.9.0) (2025-11-24)


### Features

* add service unavailable retryable ([983774b](https://github.com/zirkelc/ai-retry/commit/983774b1337626689e02e2bf80044dd7361a36b3))

## [0.8.0](https://github.com/zirkelc/ai-retry/compare/v0.7.0...v0.8.0) (2025-11-08)


### Features

* support timeouts in retry attempts ([7c45eda](https://github.com/zirkelc/ai-retry/commit/7c45edafc19be0c3158e128abe7ded5bb6a8308e))

## [0.7.0](https://github.com/zirkelc/ai-retry/compare/v0.6.0...v0.7.0) (2025-10-25)


### Features

* support static retryables with options ([7fafd30](https://github.com/zirkelc/ai-retry/commit/7fafd3035ec3e96078a6a07cf533a82f60de7280))

## [0.6.0](https://github.com/zirkelc/ai-retry/compare/v0.5.1...v0.6.0) (2025-10-23)


### Features

* override `providerOptions` ([67db0e1](https://github.com/zirkelc/ai-retry/commit/67db0e1660174f989956f67b71affb4286154b6c))

## [0.5.1](https://github.com/zirkelc/ai-retry/compare/v0.5.0...v0.5.1) (2025-10-23)


### Bug Fixes

* remove model param for `retryAfterDelay` ([53ecd26](https://github.com/zirkelc/ai-retry/commit/53ecd260f46cb6e3672964ce8098137912cdb1f2))

## [0.5.0](https://github.com/zirkelc/ai-retry/compare/v0.4.1...v0.5.0) (2025-10-21)


### Features

* support exponential backoff for every retryable ([74c9770](https://github.com/zirkelc/ai-retry/commit/74c9770cc89bbf23a353cfe5e2fae1dd1b7d5458))


### Bug Fixes

* rename RetryModel to Retry ([fa96bb3](https://github.com/zirkelc/ai-retry/commit/fa96bb3cd74185d29f68f6370be786bdf5b06083))

## [0.4.1](https://github.com/zirkelc/ai-retry/compare/v0.4.0...v0.4.1) (2025-10-18)


### Bug Fixes

* export utils ([c734bb7](https://github.com/zirkelc/ai-retry/commit/c734bb77eab5cec3d3e014bf008e097b3d00dc2a))

## [0.4.0](https://github.com/zirkelc/ai-retry/compare/v0.3.0...v0.4.0) (2025-10-16)


### Features

* retry with exponential backoff and retry-after headers ([098e50f](https://github.com/zirkelc/ai-retry/commit/098e50fed478c8ee34d6ec6729056b9008be9c53))


### Bug Fixes

* calculate backoff factor ([c0b2d00](https://github.com/zirkelc/ai-retry/commit/c0b2d003d22932f050b7dd8e5238f509e3430bf5))
* up pnpm &gt; v10 ([eabb07e](https://github.com/zirkelc/ai-retry/commit/eabb07edc6ef1fdbf18148c46f3a46f19f0e0ebf))

## [0.3.0](https://github.com/zirkelc/ai-retry/compare/v0.2.0...v0.3.0) (2025-10-15)


### Features

* support retry delays ([d10ee7a](https://github.com/zirkelc/ai-retry/commit/d10ee7a733cd90dc5a67a5068d4ac973ae99765a))


### Bug Fixes

* export types ([5a9f664](https://github.com/zirkelc/ai-retry/commit/5a9f664e64416a72a9e1f527e54fb64f6f61395e))
* timing test ([74e3b2e](https://github.com/zirkelc/ai-retry/commit/74e3b2e0db787a6bd9ecc0846817ae2aed92b2d1))

## [0.2.0](https://github.com/zirkelc/ai-retry/compare/v0.1.1...v0.2.0) (2025-10-06)


### Features

* support embedding models ([9b88e3c](https://github.com/zirkelc/ai-retry/commit/9b88e3c538bf64cae1bd55a0144bf710564261f9))

## [0.1.1](https://github.com/zirkelc/ai-retry/compare/v0.1.0...v0.1.1) (2025-10-03)


### Bug Fixes

* double retry error ([76106e4](https://github.com/zirkelc/ai-retry/commit/76106e40c750b7dfd57798d37ca67801dc214424))
