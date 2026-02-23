# Eval Readability -- Human Test Plan

## Prerequisites
- Working checkout of branch `bpowers/eval-readability` at commit `f684452`
- Node.js installed with ESM support
- `npm install` completed in `/Users/bpowers/src/sd-ai`
- Automated tests passing: `NODE_OPTIONS="--experimental-vm-modules" npx jest tests/evals/ --no-coverage` (expect 9 suites, 169 tests, 0 failures)

## Phase 1: Structural Verification of Factory Pattern Elimination

| Step | Action | Expected |
|------|--------|----------|
| 1 | Open `evals/categories/qualitativeTranslation.js`. Search the file for `relationshipEqualityComparatorGenerator`. | No results found. The old factory function is absent. |
| 2 | In the same file, locate the `relationshipMatches` function (line 263). Verify it is declared as `const relationshipMatches = function(a, b)` taking exactly two arguments. | Function signature takes exactly two arguments `(a, b)` and does not return another function. Body compares `a.from.toLowerCase()` to `b.from.toLowerCase()` and `a.to.toLowerCase()` to `b.to.toLowerCase()`. |
| 3 | In the same file, locate the call sites for `relationshipMatches`. Check lines 277-278. | Line 277: `cleanedSortedAI.some((ai) => relationshipMatches(element, ai))`. Line 278: `sortedGroundTruth.some((gt) => relationshipMatches(element, gt))`. Both use explicit lambdas passing two arguments. |
| 4 | Open `evals/categories/quantitativeTranslation.js`. Search for `stockEqualityGTComparatorGenerator` and `stockEqualityAIComparatorGenerator`. | No results found for either name. Both old factory functions are absent. |
| 5 | In the same file, locate `stockNameMatches` (line 186). Verify it is a single plain two-argument function. | Function signature is `const stockNameMatches = function(a, b)` taking exactly two arguments. Body delegates to `compareNames(a.name, b.name)`. |
| 6 | Check the call sites for `stockNameMatches` at lines 196-197 and 225. | Line 196: `sortedAIStocks.some((aiStock) => stockNameMatches(aiStock, element))`. Line 197: `sortedTruthStocks.some((gtStock) => stockNameMatches(element, gtStock))`. Line 225: `sortedAIStocks.find((aiStock) => stockNameMatches(aiStock, groundTruthStock))`. Each uses explicit lambda with correct argument order. |

## Phase 2: Structural Verification of run.js Flattening

| Step | Action | Expected |
|------|--------|----------|
| 1 | Open `evals/runHelpers.js`. Verify that `loadTestsForEngine`, `loadCategoryTests`, and `buildTestEntry` exist as named exported functions. | All three are defined as named `function` declarations and exported in the `export` block. |
| 2 | Open `evals/run.js`. Verify that the triple-nested `map().map().map()` pattern has been replaced. | Lines 96-117 show that `run.js` calls `loadCategoryTests` and `loadTestsForEngine` from the imported `runHelpers.js` module. No inline test-entry construction or inline group filtering logic exists in `run.js`. |
| 3 | Confirm `run.js` imports the helpers from `runHelpers.js`. | Lines 20-27 show the multi-line import of functions and constants from `./runHelpers.js`. |

## Phase 3: Descriptive Variable Names

| Step | Action | Expected |
|------|--------|----------|
| 1 | In `evals/categories/qualitativeTranslation.js`, locate the `.map()` callbacks that iterate over `fromAI` (line 268) and build `sortedGroundTruth` (line 273). Check parameter names. | Line 268: `fromAI.map((relationship) => {...})`. Line 273: `groundTruth.map((relationship) => {...})`. Both use `relationship`, not single-letter `r`. |
| 2 | Search the same file for any single-letter variable `r` in map/forEach callbacks on `fromAI`. | No single-letter `r` in those callbacks. The only `r` is in `stringifyRelationship(r)` (line 249), which is acceptable as an internal helper parameter. |
| 3 | In `evals/run.js`, check line 104. | `Object.entries(experiment.categories).map(async ([categoryName, filter]) => ...)`. Uses `categoryName`, not `c`. |
| 4 | In `evals/runHelpers.js`, check line 36. | `Object.entries(allTests).flatMap(([categoryName, groups]) => {...})`. Uses `categoryName`, not `c`. |
| 5 | Verify short names preserved where appropriate: in `evals/categories/conformance.js`, check the `.map()` at line 288. | `requirements.variables.map((v) => { return v.toLowerCase() })`. Uses `v` which is acceptable in context of `requirements.variables`. |
| 6 | In `evals/categories/quantitativeTranslation.js`, check `.forEach()` at lines 54 and 74. | `stock.inflows.forEach((f)=> {...})` and `stock.outflows.forEach((f)=> {...})`. Uses `f` which is acceptable given containing collection names `inflows`/`outflows`. |

## Phase 4: Modern JS Idioms

| Step | Action | Expected |
|------|--------|----------|
| 1 | In `evals/runHelpers.js`, search for `indexOf`. | No results. The file does not contain `indexOf`. |
| 2 | Check line 20 of `runHelpers.js` for the group inclusion check. | `filter.includes(groupName)` -- uses the modern `.includes()` method. |

## Phase 5: Evaluation Schema Integration

| Step | Action | Expected |
|------|--------|----------|
| 1 | In `evals/categories/qualitativeTranslation.js`, verify import of `validateEvaluationResult` and its use in the return statement. | Line 15: `import { validateEvaluationResult } from '../evaluationSchema.js';`. Line 308: `return validateEvaluationResult(failures);`. |
| 2 | In `evals/categories/quantitativeTranslation.js`, verify import and return. | Line 14: `import { validateEvaluationResult } from '../evaluationSchema.js';`. Line 275: `return validateEvaluationResult(failures);`. |
| 3 | In `evals/categories/conformance.js`, verify import and return. | Line 30: `import { validateEvaluationResult } from '../evaluationSchema.js';`. Line 328: `return validateEvaluationResult(fails);`. |

## Phase 6: Function Signatures Unchanged

| Step | Action | Expected |
|------|--------|----------|
| 1 | In `evals/categories/qualitativeTranslation.js`, check the `evaluate` export at line 245. | `export const evaluate = function(generatedResponse, groundTruth)`. Two parameters, same as pre-refactoring. |
| 2 | In `evals/categories/quantitativeTranslation.js`, check the `evaluate` export at line 172. | `export const evaluate = function(generatedResponse, groundTruth)`. Two parameters, same as pre-refactoring. |
| 3 | In `evals/categories/conformance.js`, check the `evaluate` export at line 281. | `export const evaluate = function(generatedResponse, requirements)`. Two parameters, same as pre-refactoring. |

## End-to-End: Full Eval Pipeline Integrity

| Step | Action | Expected |
|------|--------|----------|
| 1 | Run the full test suite: `NODE_OPTIONS="--experimental-vm-modules" npx jest --no-coverage` from project root. | All test suites pass, including pre-existing tests for categories not touched by the refactoring. |
| 2 | If an experiment config file is available, run `node evals/run.js --experiment <config.json>` and verify the run completes without errors. | The eval harness loads categories, applies rate limits, and runs tests without import errors or runtime exceptions from the refactored modules. |

## Traceability

| Acceptance Criterion | Automated Test | Manual Step |
|----------------------|----------------|-------------|
| AC1.1 | -- | Phase 1, steps 1-3 |
| AC1.2 | -- | Phase 1, steps 4-6 |
| AC1.3 | `qualitativeTranslation.test.js` line 24, mutation-safety block | -- |
| AC1.4 | `quantitativeTranslation.test.js` line 500, mutation-safety block | -- |
| AC2.1 | -- | Phase 2, steps 1-3 |
| AC2.2 | `run.test.js` lines 59-122 | -- |
| AC2.3 | `run.test.js` lines 13-55 | -- |
| AC3.1 | -- | Phase 3, steps 1-2 |
| AC3.2 | -- | Phase 3, steps 3-4 |
| AC3.3 | -- | Phase 3, steps 5-6 |
| AC4.1 | -- | Phase 4, steps 1-2 |
| AC5.1 | `qualitativeTranslation.test.js` lines 182-231 | -- |
| AC5.2 | `quantitativeTranslation.test.js` lines 587-618 | -- |
| AC5.3 | `conformance.test.js` lines 550-569 | -- |
| AC5.4 | All three category test files, mutation-safety blocks | -- |
| AC6.1 | `evaluationSchema.test.js` lines 4-33 | -- |
| AC6.2 | -- | Phase 5, steps 1-3 |
| AC6.3 | `evaluationSchema.test.js` lines 35-49 | -- |
| AC6.4 | `evaluationSchema.test.js` lines 52-86 | -- |
| AC7.1 | `evaluationSchema.test.js` all describe blocks | -- |
| AC7.2 | All three category test files have `input mutation safety` blocks | -- |
| AC7.3 | `run.test.js` lines 12-244 | -- |
| AC8.1 | Full suite: 9 suites, 169 tests, 0 failures | -- |
| AC9.1 | -- | Phase 6, steps 1-3 |
| AC9.2 | Existing tests + AC5.4 idempotency tests in all three files | -- |
