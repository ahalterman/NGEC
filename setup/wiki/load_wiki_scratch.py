file = "enwiki-latest-pages-articles.xml.bz2"
dump = mwxml.Dump.from_file(bzopen(file, "r"))

results = []
title_list = ['Anil Deshmukh', 'Mamata Banerjee', 'Sameer Wankhede', 'Brasilia', 'Kyle Rittenhouse', 'Ahmad Massoud', 'Ariel Henry', 'Augusto Aras',
'Geneva', 'Beirut']

for n, page in tqdm(enumerate(dump), total=100):
    title = page.title
    if title not in title_list:
        continue
    text = next(page).text
    r = parse_wiki_article(page, title, text) 
    results.append(r)
    print(title)
    #if n > 100:
    #    break
    
    , next(page).text)
    parse_wiki_article()



raw = """{{Short description|Ancient Greek city in Anatolia}}\n{{Use dmy dates|date=April 2020}}\n{{Infobox ancient site\n|name = Anazarbus\n|native_name = Anavarza {{in lang|tr}}\n|alternate_name = Caesarea, Justinopolis\n|image = Anavarza_Triumphal_arch_in_Anazarbus_2754.jpg\n|alt = \n|caption = The triumphal arch of Anazarbus was later converted to the city\'s South Gate.\n|map_type = Turkey\n|map_alt = \n|map_size = 270\n|coordinates = {{coord|37|15|50|N|35|54|20|E|display=inline,title}}\n|location = [[Adana Province]], Turkey\n|region = [[Cilicia]]\n|type = Settlement\n|part_of = \n|length = \n|width = \n|area = \n|height = \n|builder = \n|material = \n|built = \n|abandoned = \n|epochs = <!-- actually displays as "Periods" -->\n|cultures = \n|dependency_of = \n|occupants = \n|event = \n|excavations = \n|archaeologists = \n|condition = \n|ownership = \n|management = \n|public_access = \n|website = <!-- {{URL|example.com}} -->\n|notes = \n}}\n\n[[File:Anazarbe_vue_générale_1.jpg|thumb|right|300px|General view of the site]]\n[[Image:Anazarbus clikya west gate and anvarza castle.JPG|thumb|right|200px|Anazarbus West Gate]]\n\'\'\'Anazarbus \'\'\' ({{lang-grc|Ἀναζαρβός}}, medieval \'\'\'Ain Zarba\'\'\'; modern \'\'\'Anavarza\'\'\'; {{lang-ar|عَيْنُ زَرْبَة}}) was an ancient [[Cilicia]]n city. Under the late Roman Empire, it was the capital of [[Cilicia Secunda]]. [[Roman emperor]] [[Justinian I]] rebuilt the city in 527 after a strong earthquake hit it. It was destroyed in 1374 by the forces of [[Mamluk Empire]], after their conquest of Armenia.\n\n"""
raw = """'{{Short description|Ethnic group in Japan and Russia}}\n{{For|the ethnic group of Western China|Äynu people}}\n{{Use mdy dates|date=April 2020}}\n{{Infobox ethnic group\n| group            = Ainu\n| image            = File:Ainu Marriage.jpg \n| image_alt        = \n| caption          = Ainu at a traditional marriage ceremony in [[Hokkaido]].\n| population       = {{plainlist|\n* 25,000\n* (Japanese government estimate, 2002)\n* ≥200,000\n* (Unofficial estimate)<ref name="Poisson, B 2002, p.5">{{cite book|last=Poisson|first=Barbara Aoki|year=2002|title=The Ainu of Japan|publisher=Lerner Publications|location=Minneapolis|page=[https://archive.org/details/ainuofjapan00pois/page/5 5]|isbn=978-0-82254-176-9|url-access=registration|url=https://archive.org/details/ainuofjapan00pois/page/5}}</ref>"""
raw = """"{{short description|Political philosophy and movement}}\n{{other uses}}\n{{redirect2|Anarchist|Anarchists|other uses|Anarchist (disambiguation)}}\n{{distinguish|Anarchy}}\n{{pp-semi-indef}}\n{{good article}}\n{{use British English|date=August 2021}}\n{{use dmy dates|date=August 2021}}\n{{anarchism sidebar}}\n{{basic forms of government}}\n'"""
re.findall("\{\{Short description\|(.+?)\}\}", raw)[0]


raw = """thumb|Main amethyst-producing countries\n\nAmethyst is a violet variety of quartz. The name comes from the Koine Greek αμέθυστος amethystos from α- a-, "not" and μεθύσκω (Ancient Greek)"""
raw = """\n\nAmethyst is a violet variety of quartz. The name comes from the Koine Greek αμέθυστος amethystos from α- a-, "not" and μεθύσκω (Ancient Greek)"""
re.sub("^thumb\|.+?\n", "", raw)