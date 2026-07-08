const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const FIGURES_DIR = path.join(__dirname, 'report_figures');
if (!fs.existsSync(FIGURES_DIR)) {
    fs.mkdirSync(FIGURES_DIR, { recursive: true });
}

async function capture() {
    console.log("Starting Puppeteer browser...");
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });

    try {
        // 1. Landing Page (Figure 8.7 Web Interface Screenshot)
        console.log("Navigating to landing page...");
        await page.goto('http://127.0.0.1:5000', { waitUntil: 'networkidle2' });
        await new Promise(resolve => setTimeout(resolve, 1500)); // wait for transitions
        const landingPath = path.join(FIGURES_DIR, 'fig_8_7_web_interface.png');
        await page.screenshot({ path: landingPath });
        console.log(`Saved: ${landingPath}`);

        // 2. Perform scan for @susanrivera (Figure 8.8 Detection Results Screenshot)
        console.log("Performing search for @susanrivera...");
        await page.focus('#search-input');
        await page.keyboard.type('susanrivera');
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Click search and wait for navigation/results
        console.log("Submitting query...");
        await Promise.all([
            page.click('.btn-initialize'),
            page.waitForNavigation({ waitUntil: 'networkidle2' })
        ]);
        
        await new Promise(resolve => setTimeout(resolve, 2000)); // wait for progress bar animation
        const resultsPath = path.join(FIGURES_DIR, 'fig_8_8_detection_results.png');
        await page.screenshot({ path: resultsPath });
        console.log(`Saved: ${resultsPath}`);
 
        // 3. Admin Login & Dashboard (Figure 8.9 Admin Dashboard Screenshot)
        console.log("Navigating to Admin Login...");
        await page.goto('http://127.0.0.1:5000/admin/login', { waitUntil: 'networkidle2' });
        await page.focus('#username');
        await page.keyboard.type('admin');
        await page.focus('#password');
        await page.keyboard.type('admin123');
        
        console.log("Logging in as admin...");
        await Promise.all([
            page.click('.btn-login'),
            page.waitForNavigation({ waitUntil: 'networkidle2' })
        ]);
        
        await new Promise(resolve => setTimeout(resolve, 2000)); // wait for ChartJS to render
        const adminPath = path.join(FIGURES_DIR, 'fig_8_9_admin_dashboard.png');
        await page.screenshot({ path: adminPath });
        console.log(`Saved: ${adminPath}`);
 
        // 4. KNN Anomaly Detection Graph (Figure 8.10 KNN Anomaly Detection Graph)
        console.log("Navigating to KNN visualizer for susanrivera...");
        await page.goto('http://127.0.0.1:5000/admin/dashboard?query=susanrivera&platform=twitter', { waitUntil: 'networkidle2' });
        await new Promise(resolve => setTimeout(resolve, 2500)); // wait for KNN plot rendering and nearest lines
        
        const knnPath = path.join(FIGURES_DIR, 'fig_8_10_knn_anomaly_graph.png');
        await page.screenshot({ path: knnPath });
        console.log(`Saved: ${knnPath}`);

        console.log("\nAll figures captured successfully!");
        
        // Also copy files to the brain artifact directory so the user sees them in artifacts
        const artifactDir = '/Users/nikhitagp/.gemini/antigravity/brain/ce02ed03-be32-4bcb-a9f8-ea6c21810276';
        if (fs.existsSync(artifactDir)) {
            const destDir = path.join(artifactDir, 'report_figures');
            if (!fs.existsSync(destDir)) {
                fs.mkdirSync(destDir, { recursive: true });
            }
            fs.copyFileSync(landingPath, path.join(destDir, 'fig_8_7_web_interface.png'));
            fs.copyFileSync(resultsPath, path.join(destDir, 'fig_8_8_detection_results.png'));
            fs.copyFileSync(adminPath, path.join(destDir, 'fig_8_9_admin_dashboard.png'));
            fs.copyFileSync(knnPath, path.join(destDir, 'fig_8_10_knn_anomaly_graph.png'));
            console.log(`Copied all screenshots to brain artifacts directory: ${destDir}`);
        }

    } catch (err) {
        console.error("Error during capture:", err);
    } finally {
        await browser.close();
    }
}

capture();
