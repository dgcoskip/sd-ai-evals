import fs from 'fs';

function parseResults(filePath) {
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
}

function generateLeaderboard(results) {
    const leaderboard = results.map(result => ({
        name: result.name,
        score: result.score
    }));

    leaderboard.sort((a, b) => b.score - a.score);
    return leaderboard;
}

function printLeaderboard(leaderboard) {
    console.log("Leaderboard:");
    leaderboard.forEach((entry, index) => {
        console.log(`${index + 1}. ${entry.name}: ${entry.score}`);
    });
}

const results = parseResults('full_results.json');
const leaderboard = generateLeaderboard(results);
printLeaderboard(leaderboard);
