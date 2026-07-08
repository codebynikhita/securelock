const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-blink-features=AutomationControlled'
        ]
    });
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36');
    await page.setViewport({ width: 1280, height: 800 });
    
    try {
        console.log("Navigating to Imginn...");
        await page.goto('https://imginn.com/__nikhita__09/', { waitUntil: 'networkidle2', timeout: 15000 });
        
        console.log("Waiting 5 seconds...");
        await new Promise(r => setTimeout(r, 6000));
        
        const data = await page.evaluate(() => {
            return {
                title: document.title,
                url: window.location.href,
                body: document.body.innerText.substring(0, 1500)
            };
        });
        console.log("Result:\n", JSON.stringify(data, null, 2));
    } catch (e) {
        console.error("Error:", e.message);
    } finally {
        await browser.close();
    }
})();
