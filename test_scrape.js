const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36');
    
    try {
        console.log("Navigating to Google Search...");
        await page.goto('https://www.google.com/search?q=site:instagram.com/__nikhita__09', { waitUntil: 'networkidle2', timeout: 10000 });
        
        const data = await page.evaluate(() => {
            return document.body.innerText.substring(0, 2000);
        });
        console.log("Scraped Google Body Text:\n", data);
    } catch (e) {
        console.error("Error:", e.message);
    } finally {
        await browser.close();
    }
})();
