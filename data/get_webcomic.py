import requests, bs4

firstp = 1
lastp = 8130

imglist = []

for x in range(firstp, lastp+1):
    print(x)
    url = "https://www.homestuck.com/story/" + str(x)
    try:
        page = requests.get(url)
        page.raise_for_status()
    except requests.exceptions.RequestException:
        continue  #some numbers are missing from 1-8130, if the link 404s skip it

    soup = bs4.BeautifulSoup(page.text, 'html.parser')
    images = soup.find_all('img', class_="mar-x-auto disp-bl")

    for count, image in enumerate(images, 1):
        imgurl = image['src']
        
        #handle local reference
        if imgurl[0] == '/':
            imgurl = "https://www.homestuck.com" + imgurl 

        response = requests.get(imgurl)
        if response.status_code == 200:
            with open("./screens/img/" + str(x) + "_" + str(count) + "." + imgurl.split(".")[-1], 'wb') as f:
                f.write(response.content) 

    title = soup.find('h2', class_='type-hs-header')
    if title:
        title = title.text.strip()
    else:
        title = ""

    text_content = ""

    story_content = soup.find('p', class_='o-story_text')
    if story_content:
        text_content += story_content.get_text(separator='\n', strip=True)
    
    chat_content = soup.find('div', class_='o_chat-container')
    if chat_content:
        text_content += chat_content.get_text(separator='\n', strip=True)

    next_link = soup.find('div', class_='o_story-nav')
    if next_link:
        next_link_text = next_link.find('a')
        if next_link_text:
            next_link_text = next_link_text.text.strip()
        else:
            next_link_text = ""
    else:
        next_link_text = ""

    print(title)
    print()
    with open("./screens/img/" + str(x) + "_textcontent.txt", 'w', encoding='utf-8') as f:
        f.write(f"Title: {title}\n\n")
        f.write(f"Text Content:\n{text_content}\n\n")
        f.write(f"Next Link Text: {next_link_text}")
