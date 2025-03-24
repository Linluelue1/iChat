import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import time


def get_links(url):
    """获取包含'听力原文'的链接"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for a in soup.find_all('a'):
            link_text = a.text.strip()
            if '答案解析' in link_text:
                href = a.get('href')
                absolute_url = urljoin(url, href)
                links.append(absolute_url)
        return list(set(links))  # 去重
    except Exception as e:
        print(f"获取链接失败: {e}")
        return []


def get_article_content(url):
    """获取文章内容"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', {'itemprop': 'articleBody'})
        if content_div:
            # 清理不需要的标签
            for tag in content_div(['script', 'style', 'iframe', 'img']):
                tag.decompose()
            return content_div.get_text(separator='\n', strip=True)
        return None
    except Exception as e:
        print(f"获取内容失败[{url}]: {e}")
        return None


def main():
    start_url = "http://www.laokaoya.com/48685.html"

    # 获取所有目标链接
    print("正在收集听力原文链接...")
    target_links = get_links(start_url)
    print(f"找到 {len(target_links)} 个链接")

    # 遍历所有链接获取内容
    results = []
    for i, link in enumerate(target_links, 1):
        print(f"正在处理第 {i}/{len(target_links)} 个链接: {link}")
        content = get_article_content(link)
        if content:
            results.append({
                'url': link,
                'content': content
            })
        time.sleep(1)  # 礼貌性延迟

    # 保存结果
    if results:
        with open('../ListeningChat/听力答案合集.txt', 'w', encoding='utf-8') as f:
            for item in results:
                f.write(f"URL: {item['url']}\n")
                f.write(f"内容:\n{item['content']}\n")
                f.write("\n" + "=" * 50 + "\n\n")
        print(f"成功保存 {len(results)} 篇听力原文到 听力答案合计.txt")
    else:
        print("未找到有效内容")


if __name__ == "__main__":
    main()