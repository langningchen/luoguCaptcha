// ==UserScript==
// @name         Captcha predict
// @namespace    https://github.com/langningchen
// @version      v0.0.1
// @description  Predict the captcha of Luogu
// @author       langningchen
// @match        *://www.luogu.com.cn/*
// @icon         https://www.luogu.com.cn/favicon.ico
// @grant        GM_xmlhttpRequest
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
        return new Promise((resolve) => {
            GM_xmlhttpRequest({
                method: 'POST',
                url: predictServer,
                headers: {
                    'Content-Type': 'application/json'
                },
                data: JSON.stringify({ image: dataURL }),
                onload: (response) => {
                    resolve(JSON.parse(response.responseText).prediction);
                }
            });
        });
    };

    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            const nodes = [];
            if (mutation.type === 'childList') {
                nodes.push(...mutation.addedNodes);
            }
            else if (mutation.type === 'attributes') {
                nodes.push(mutation.target);
            }
            for (const node of nodes) {
                if (node.nodeName === 'IMG' && node.src.includes('captcha')) {
                    const imageElement = node;
                    if (location.pathname === '/image') {
                        const inputElement = imageElement.parentElement.nextSibling;
                        const submitElement = inputElement.parentElement.nextSibling.firstElementChild;
                        imageElement.onload = async () => {
                            inputElement.value = await predict(imageElement);
                            submitElement.click();
                        };
                    }
                    else if (location.pathname === '/auth/login') {
                        const inputElement = imageElement.parentElement.previousElementSibling.firstElementChild
                        imageElement.onload = async () => {
                            inputElement.value = await predict(imageElement);
                            inputElement.dispatchEvent(new Event('input'));
                        };
                    }
                    else if (location.pathname.startsWith('/discuss/')) {
                        const inputElement = imageElement.parentElement.previousElementSibling;
                        imageElement.onload = async () => {
                            inputElement.value = await predict(imageElement);
                            inputElement.dispatchEvent(new Event('input'));
                        };
                    }
                    else {
                        alert('Unknown page');
                    }
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });
})();
