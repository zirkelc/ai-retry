# Changelog

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
