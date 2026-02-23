const BASELINE_TOKEN_USAGE = 3000;
const TOKENS_PER_MINUTE = 30_000;
const REQUESTS_PER_MINUTE = 400;

function applyDefaultLimits(engineConfig) {
  engineConfig.limits = engineConfig.limits || {};
  engineConfig.limits.tokensPerMinute =
    engineConfig.limits.tokensPerMinute || TOKENS_PER_MINUTE;
  engineConfig.limits.requestsPerMinute =
    engineConfig.limits.requestsPerMinute || REQUESTS_PER_MINUTE;
  engineConfig.limits.baselineTokenUsage =
    engineConfig.limits.baselineTokenUsage || BASELINE_TOKEN_USAGE;
}

function loadCategoryTests(groups, filter) {
  if (filter === true) return groups;
  if (filter === false) return {};
  return Object.fromEntries(
    Object.entries(groups).filter(([groupName, _]) => {
      return filter.includes(groupName);
    })
  );
}

function buildTestEntry(test, engineConfig, engineConfigName, categoryName, groupName) {
  return {
    engineConfig,
    engineConfigName,
    category: categoryName,
    group: groupName,
    testParams: test,
  };
}

function loadTestsForEngine(allTests, engineConfig, engineConfigName) {
  return Object.entries(allTests).flatMap(([categoryName, groups]) => {
    return Object.entries(groups).flatMap(([groupName, tests]) => {
      return tests.map((test) => buildTestEntry(test, engineConfig, engineConfigName, categoryName, groupName));
    });
  });
}

export {
  BASELINE_TOKEN_USAGE,
  TOKENS_PER_MINUTE,
  REQUESTS_PER_MINUTE,
  applyDefaultLimits,
  loadCategoryTests,
  buildTestEntry,
  loadTestsForEngine,
};
