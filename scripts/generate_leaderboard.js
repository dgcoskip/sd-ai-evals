import fs from 'fs';
import readline from 'readline';

function parseResults(filePath) {
    if (!fs.existsSync(filePath)) {
        console.error(`Error: File not found - ${filePath}`);
        process.exit(1);
    }
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

function listMatchingFiles(pattern) {
    return fs.readdirSync('.').filter(file => file.match(pattern));
}

function promptUserForFile(files) {
    return new Promise((resolve) => {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        console.log("Available result files:");
        files.forEach((file, index) => {
            console.log(`${index + 1}: ${file}`);
        });

        rl.question('Select a file by number: ', (answer) => {
            const index = parseInt(answer, 10) - 1;
            if (index >= 0 && index < files.length) {
                resolve(files[index]);
            } else {
                console.error("Invalid selection.");
                process.exit(1);
            }
            rl.close();
        });
    });
}

async function main() {
    // print the time ai!
    const files = listMatchingFiles(/^.*full_results\.json$/);
    if (files.length === 0) {
        console.error("No matching files found.");
        process.exit(1);
    }
    const selectedFile = await promptUserForFile(files);
    const results = parseResults(selectedFile);
    const leaderboard = generateLeaderboard(results);
    printLeaderboard(leaderboard);
}

main();
const leaderboard = generateLeaderboard(results);
printLeaderboard(leaderboard);
