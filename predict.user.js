// ==UserScript==
// @name        Captcha predict
// @namespace   https://github.com/langningchen
// @version     v0.0.4
// @description Predict the captcha of Luogu
// @author      langningchen
// @match       *://www.luogu.com.cn/*
// @icon        https://www.luogu.com.cn/favicon.ico
// @grant       GM_xmlhttpRequest
// @updateURL   https://github.com/langningchen/luoguCaptcha/raw/refs/heads/main/predict.user.js
// @downloadURL https://github.com/langningchen/luoguCaptcha/raw/refs/heads/main/predict.user.js
// ==/UserScript==

const predictServer = 'https://luogu.cyezoi.com';

(() => {
    window.predict = async (imageElement) => {
        const canvas = document.createElement('canvas');
        canvas.width = imageElement.width;
        canvas.height = imageElement.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0);
        const dataURL = canvas.toDataURL('image/jpeg').split(',')[1];
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: 'POST',
                url: predictServer,
                headers: {
                    'Content-Type': 'application/json'
                },
                data: JSON.stringify({ image: dataURL }),
                onload: (response) => {
                    try {
                        const result = JSON.parse(response.responseText);
                        resolve(result.prediction);
                    } catch (e) {
                        console.error('Failed to parse prediction response:', e);
                        reject(new Error('Failed to parse prediction response.'));
                    }
                },
                onerror: (response) => {
                    console.error('GM_xmlhttpRequest failed:', response);
                    reject(new Error('GM_xmlhttpRequest failed.'));
                }
            });
        });
    };

    const findCaptchaAndFill = (element) => {
        if (!element || element.nodeType !== Node.ELEMENT_NODE) return;

        const images = element.querySelectorAll('img[src*="captcha"]');
        if (images.length === 0 && element.nodeName === 'IMG' && element.src.includes('captcha')) {
            // 如果传入的元素本身就是验证码图片
            processImage(element);
        } else {
            // 如果传入的元素包含验证码图片
            images.forEach(processImage);
        }
    };

    const processImage = async (imageElement) => {
        const inputElement = findCaptchaInput();
        if (!inputElement) {
            console.warn("找不到验证码输入框");
            return;
        }

        // 等待图片加载完成再进行预测
        if (imageElement.complete && imageElement.naturalWidth !== 0) {
            try {
                inputElement.value = await predict(imageElement);
                dispatchInputEvents(inputElement);
            } catch (e) {
                console.error("预测验证码失败:", e);
            }
        } else {
            imageElement.onload = async () => {
                try {
                    inputElement.value = await predict(imageElement);
                    dispatchInputEvents(inputElement);
                } catch (e) {
                    console.error("预测验证码失败:", e);
                }
            };
            imageElement.onerror = () => {
                console.error("验证码图片加载失败:", imageElement.src);
            };
        }
    };

    const findCaptchaInput = () => {
        const inputs = document.querySelectorAll('input[placeholder*="验证码"], input[id*="captcha"], input[name*="captcha"]');
        for (const input of inputs) {
            // 确保找到的是可见的、可输入的输入框
            if (input.offsetWidth > 0 || input.offsetHeight > 0) {
                return input;
            }
        }
        return null;
    };

    const dispatchInputEvents = (inputElement) => {
        inputElement.dispatchEvent(new Event('input', { bubbles: true }));
        inputElement.dispatchEvent(new Event('change', { bubbles: true }));
    };

    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                mutation.addedNodes.forEach(findCaptchaAndFill);
            } else if (mutation.type === 'attributes' && mutation.target.nodeName === 'IMG' && mutation.target.src.includes('captcha')) {
                findCaptchaAndFill(mutation.target);
            }
        }
    });

    // 初始页面加载时检查一次
    findCaptchaAndFill(document.body);

    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src']
    });
})();
