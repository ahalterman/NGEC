import pytest
import datetime

def test_nat1(ag):
    cc, text = ag.search_nat("Great Britain")
    assert cc == "GBR"
    assert text == ""

def test_nat2(ag):
    cc, text = ag.search_nat("Republic of China")
    assert cc == "TWN"
    assert text == ""

def test_nat3(ag):
    cc, text = ag.search_nat("People's Republic of China")
    assert cc == "CHN"
    assert text == ""

def test_nat4(ag):
    cc, text = ag.search_nat("Saudi Arabia")
    assert cc == "SAU"
    assert text == ""

def test_nat5(ag):
    cc, text = ag.search_nat("four Saudi men")
    assert cc == "SAU"
    assert text == "four  men"

def test_nat5(ag):
    cc, text = ag.search_nat("an American destroyer")
    assert cc == "USA"
    assert text == "destroyer"

def test_nat6(ag):
    cc, text = ag.search_nat("a US destroyer")
    assert cc == "USA"
    assert text == "destroyer"


def test_shorttext_1(ag):
    code = ag.short_text_to_agent("soldier")
    assert code['code_1'] == "MIL"
    assert code['pattern'] == "soldier"
    assert code['country'] == ""

def test_shorttext_2(ag):
    code = ag.short_text_to_agent("soldiers")
    assert code['code_1'] == "MIL"
    assert code['pattern'] != "soldiers"
    assert code['country'] == ""

def test_shorttext_3(ag):
    code = ag.short_text_to_agent("American warplanes")
    assert code['code_1'] == "MIL"
    assert code['pattern'] == "warplane"
    assert code['country'] == "USA"

def test_shorttext_4(ag):
    code = ag.short_text_to_agent("retired vice president")
    assert code['code_1'] == "ELI"

cop_list = ["cyber police", "crowd control police", "paris railway police", "seoul metropolitan police"]
@pytest.mark.nondestructive
@pytest.mark.parametrize("cop", cop_list)
def test_cop_list(ag, cop):
    code = ag.short_text_to_agent(cop)
    assert code['code_1'] == "COP"

crm_list = ["cyber criminals", "vandals", "white supremicists", "group of thieves", "motorcycle thieves", 
            "wanted persons", "smuggling gang"]
@pytest.mark.nondestructive
@pytest.mark.parametrize("crm", crm_list)
def test_crm_list(ag, crm):
    code = ag.short_text_to_agent(crm)
    assert code['code_1'] == "CRM"

med_list = ["paramedic", "medic"]
@pytest.mark.nondestructive
@pytest.mark.parametrize("med", med_list)
def test_crm_list(ag, med):
    code = ag.short_text_to_agent(med)
    assert code['code_1'] == "MED"

reb_list = ["mujahideen", "jihadis"]
@pytest.mark.nondestructive
@pytest.mark.parametrize("code", reb_list)
def test_crm_list(ag, code):
    code = ag.short_text_to_agent(code)
    assert code['code_1'] == "REB"

def test_gov_list(ag):
    cop_list = ["cyber police", "crowd control police", "paris railway police", "seoul metropolitan police"]
    for cop in cop_list:
        code = ag.short_text_to_agent(cop)
        assert code['code_1'] == "COP"

def test_shorttext_5(ag):
    code = ag.short_text_to_agent("Director of Central Intelligence")
    assert code['code_1'] == "SPY"


def test_wiki1(ag):
    wiki = ag.query_wiki("Abkhazia")
    assert wiki['title'] == "Abkhazia"

def test_wiki3(ag):
    wiki = ag.query_wiki("Angela Merkel")
    assert wiki['title'] == "Angela Merkel"  

def test_wiki4(ag):
    wiki = ag.query_wiki("Mario Abdo Benítez")
    assert wiki['title'] == "Mario Abdo Benítez" 

def test_wiki4_2(ag):
    wiki = ag.query_wiki("Mario Abdo Benitez")
    assert wiki['title'] == "Mario Abdo Benítez" 

def test_wiki5(ag):
    wiki = ag.query_wiki("Nicolas Maduro")
    assert wiki['title'] == "Nicolás Maduro"
    
def test_wiki6(ag):
    wiki = ag.query_wiki("The United Nations")
    assert wiki['title'] == "United Nations"

def test_wiki7(ag):
    wiki = ag.query_wiki("the U.N.")
    assert wiki['title'] == "United Nations"

def test_wiki8(ag):
    wiki = ag.query_wiki("Collective Security Treaty Organization")
    assert wiki['title'] == 'Collective Security Treaty Organization'

def test_wiki9(ag):
    wiki = ag.query_wiki("The Collective Security Treaty Organization")
    assert wiki['title'] == 'Collective Security Treaty Organization'

def test_wiki10(ag):
    wiki = ag.query_wiki("The North Atlantic Treaty Organization")
    assert wiki['title'] == 'NATO'

def test_wiki11(ag):
    wiki = ag.query_wiki("Kassym-Jomart Tokayev")
    assert wiki['title'] == 'Kassym-Jomart Tokayev'

def test_wiki12(ag):
    wiki = ag.query_wiki("President Kassym-Jomart Tokayev")
    assert wiki['title'] == 'Kassym-Jomart Tokayev'

def test_wiki13(ag):
    wiki = ag.query_wiki("George W. Bush")
    assert wiki['title'] == 'George W. Bush'

def test_wiki14(ag):
    wiki = ag.query_wiki("George Dubya Bush")
    assert wiki['title'] == 'George W. Bush'

def test_wiki15(ag):
    wiki = ag.query_wiki("George W.  Bush")
    assert wiki['title'] == 'George W. Bush'

def test_wiki16(ag):
    wiki = ag.query_wiki("G. W.  Bush")
    assert wiki['title'] == 'George W. Bush'

def test_wiki17(ag):
    wiki = ag.query_wiki("Ferdinand Marcos Jr.")
    assert wiki['title'] == 'Bongbong Marcos'

def test_wiki17_2(ag):
    wiki = ag.query_wiki("Ferdinand Marcos")
    assert wiki['title'] == 'Ferdinand Marcos'

def test_wiki18(ag):
    wiki = ag.query_wiki("President Duterte")
    assert wiki['title'] == 'Rodrigo Duterte'

def test_wiki19(ag):
    wiki = ag.query_wiki("Vetevendosje")
    assert wiki['title'] == 'Vetëvendosje'

### These pages were missing in the summer 2022 index.

def test_wiki20(ag):
    wiki = ag.query_wiki("Nato")
    assert wiki['title'] == 'NATO'

def test_wiki21(ag):
    wiki = ag.query_wiki("Mamata Banerjee")
    assert wiki['title'] == 'Mamata Banerjee'

def test_wiki22(ag):
    # Missing entirely from Wiki index
    wiki = ag.query_wiki("Augusto Aras")
    assert wiki['title'] == 'Augusto Aras'

def test_wiki23(ag):
    wiki = ag.query_wiki("ECOWAS")
    assert wiki['title'] == 'Economic Community of West African States'

def test_wiki24(ag):
    wiki = ag.query_wiki("Nato")
    assert wiki['title'] == 'NATO'

def test_wiki25(ag):
    # Incorrectly parsed intro para
    # 'intro_para': ' \n*\nOrbán, Viktor',
    wiki = ag.query_wiki("Viktor Orban")
    assert wiki['title'] == 'Viktor Orbán'

def test_wiki26(ag):
    wiki = ag.query_wiki("Anil Deshmukh")
    assert wiki['title'] == 'Anil Deshmukh'

def test_wiki27(ag):
    wiki = ag.query_wiki("Geneva")
    assert wiki['title'] == 'Geneva'
    assert len(wiki['intro_para']) > 30

def test_wiki28(ag):
    wiki = ag.query_wiki("Muhammadu Buhari")
    assert wiki['title'] == 'Muhammadu Buhari'
    assert len(wiki['intro_para']) > 30

def test_wiki29(ag):
    wiki = ag.query_wiki("Beirut")
    assert wiki['title'] == 'Beirut'
    assert len(wiki['intro_para']) > 30

def test_wiki30(ag):
    wiki = ag.query_wiki("Brasilia")
    assert wiki['title'] == 'Brasília'
    assert len(wiki['intro_para']) > 30

def test_wiki31(ag):
    wiki = ag.query_wiki("Warsaw")
    assert wiki['title'] == 'Warsaw'
    assert len(wiki['intro_para']) > 30

def test_wiki32(ag):
    wiki = ag.query_wiki("Dmitry Peskov")
    assert wiki['title'] == 'Dmitry Peskov'
    assert len(wiki['intro_para']) > 30

def test_wiki33(ag):
    wiki = ag.query_wiki("Kyle Rittenhouse")
    assert wiki['title'] == 'Kyle Rittenhouse'
    assert len(wiki['intro_para']) > 30

def test_wiki34(ag):
    wiki = ag.query_wiki("Jacob Zuma")
    assert wiki['title'] == 'Jacob Zuma'
    assert len(wiki['intro_para']) > 30

def test_wiki35(ag):
    wiki = ag.query_wiki("MEPs")
    assert wiki['title'] == 'Member of the European Parliament'
    assert len(wiki['intro_para']) > 30

def test_wiki36(ag):
    wiki = ag.query_wiki("Trump")
    assert wiki['title'] == 'Donald Trump'
    assert len(wiki['intro_para']) > 30

def test_wiki37(ag):
    wiki = ag.query_wiki("Juan Carlos Jobet")
    assert wiki['title'] == "Juan Carlos Jobet"
    assert len(wiki['intro_para']) > 30

def test_wiki38(ag):
    wiki = ag.query_wiki("Islamic State")
    assert wiki['title'] == "Islamic State"

def test_wiki38(ag):
    wiki = ag.query_wiki("ISIS")
    assert wiki['title'] == "Islamic State"

####

def test_nonsense_agent_resolution(ag):
    code = ag.trf_agent_match("a cat named Frank who recently obtained a ball of string")
    assert code is None

def test_nonsense2(ag):
    code = ag.trf_agent_match("a cat named Frank")
    assert code is None

def test_nonsense3(ag):
    code = ag.trf_agent_match("Frank")
    assert code is None

def test_nonsense4(ag):
    code = ag.trf_agent_match("a cat")
    assert code is None

##########

def test_igo_full(ag):
    d = {"actor": "Collective Security Treaty Organization",
        "context": "",
        "date": "Today",
        "correct_country": "IGO"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['country'] == d['correct_country'] 


    
def test_kassym(ag):
    #### POSSIBLE SOLUTION: if no hits on the first query, strip out non-ents and try again?
    d = {"actor": "President Kassym-Jomart Tokayev",
        "context": "",
        "date": "Today",
        "correct_country": "KAZ",
        "correct_code1": "GOV"
        }
    wiki = ag.query_wiki(d['actor'])
    assert wiki['title'] == 'Kassym-Jomart Tokayev'
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_kassym2(ag):
    code = ag.agent_to_code("Kassym-Jomart Tokayev", "", "today")
    assert code['country'] == "KAZ"
    assert code['code_1'] == "GOV"


def kaz1(ag):
    code = ag.agent_to_code("Kazakhstan")
    assert code['country'] == "KAZ"
    assert code['code_1'] == ""

def kaz2(ag):
    d = {"actor": "ordinary Kazakhs",
        "context": "",
        "date": "Today",
        "correct_country": "KAZ",
        "correct_code1": "CVL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def kaz3(ag): 
    d = {"actor": "Kazahks",
        "context": "",
        "date": "Today",
        "correct_country": "KAZ",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def kaz4(ag):
    d = {"actor": "former President Nursultan Nazarbayev",
        "context": "",
        "date": "Today",
        "correct_country": "KAZ",
        "correct_code1": "CVL" # "former" officials used to be ELI, now they're CVL
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
 
def test_katushya(ag):
    code = ag.agent_to_code("Katushya")
    assert code['code_1'] == "MIL"

def test_northern_fleet_1(ag):
    code = ag.agent_to_code("Northern Fleet")
    assert code['wiki'] == "Northern Fleet"
    assert code['country'] == "RUS"
    assert code['code_1'] == "MIL"

def test_full_armenia(ag):
    # TODO: split out titles and names before querying wiki?
    code = ag.agent_to_code("Armenian Prime Minister Nikol Pashinyan")
    assert code['country'] == "ARM"
    assert code['code_1'] == "GOV"

def test_armenia_title(ag):
    code = ag.agent_to_code("Armenian Prime Minister")
    assert code['country'] == "ARM"
    assert code['code_1'] == "GOV"
    
def test_armenia_name(ag):
    code = ag.agent_to_code("Nikol Pashinyan")
    assert code['country'] == "ARM"
    assert code['code_1'] == "GOV"
    
def test_mex(ag):
    d = {"actor": "Mexican authorities",
        "context": "",
        "date": "Today",
        "correct_country": "MEX",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        
def test_dea(ag):
    code = ag.agent_to_code("The DEA", "", "")
    assert code['wiki'] == 'Drug Enforcement Administration'
    assert code['country'] == "USA"
    assert code['code_1'] == "COP"

@pytest.mark.xfail(reason="Can't get this correct without context")
def test_pan(ag):
    d = {"actor": "The PAN",
        "context": "",
        "date": "Today",
        "correct_country": "MEX",
        "correct_code1": "PTY"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

@pytest.mark.skip(reason="Wiki searches currently don't use context")
def test_pan_context(ag):
    d = {"actor": "The PAN",
        "context": "Mexico",
        "date": "Today",
        "correct_country": "MEX",
        "correct_code1": "PTY"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_pri(ag):
    """
    Rare case where there's a good agent match...but it should query
    Wikipedia. Don't match if the whole thing looks like an entity?
    Or also check Wiki if it looks like an entity?
    """
    code = ag.agent_to_code("Institutional Revolutionary Party")
    assert code['wiki'] == 'Institutional Revolutionary Party'
    assert code['country'] == "MEX"
    assert code['code_1'] == "PTY"


def test_pri2(ag):
    wiki = ag.query_wiki("the PRI", "Mexico")
    assert wiki['title'] == 'Institutional Revolutionary Party'
    code = ag.agent_to_code("the PRI")
    assert code['country'] == "MEX"
    assert code['code_1'] == "PTY"

def test_pri3(ag):
    wiki = ag.query_wiki("the PRI Mexico")
    assert wiki['title'] == 'Institutional Revolutionary Party'

def test_pri_context(ag):
    wiki = ag.query_wiki("PRI", country="Mexico")
    assert wiki['title'] == 'Institutional Revolutionary Party'
    code = ag.agent_to_code("PRI", known_country="Mexico")
    assert code['country'] == "MEX"
    assert code['code_1'] == "PTY"


def test_hezbollah(ag):
    # side note: here's an interesting case of wanting two separate, non-hierarchical
    # codes, one PTY and one REB
    code = ag.agent_to_code("Hezbollah", "", "today")
    assert code['code_1'] in ["PTY", "REB"]
    assert code['country'] == "LBN"
    assert code['wiki'] == "Hezbollah"

def test_prd(ag):
    wiki = ag.query_wiki("the PRD Mexico")
    assert wiki['title'] == 'Party of the Democratic Revolution'




@pytest.mark.skip(reason="Wiki searches currently don't use context")       
def test_prd_context(ag):
    d = {"actor": "the PRD",
        "context": "Mexican",
        "date": "Today",
        "correct_country": "MEX",
        "correct_code1": "PTY"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_zetas(ag):
    wiki = ag.query_wiki("Los Zetas")
    assert wiki['title'] == 'Los Zetas'
    code = ag.agent_to_code("Los Zetas", "", "2022-01-01")
    assert code['country'] == 'MEX'
    assert code['code_1'] == 'CRM'

def test_zetas2(ag):
    """
    Another example 
    """
    wiki = ag.query_wiki("the Zetas Cartel")
    assert wiki['title'] == 'Los Zetas'
    code = ag.agent_to_code("the Zetas Cartel", "", "2022-01-01")
    assert code['country'] == "MEX"
    assert code['code_1'] == "CRM"
    
#def test_city1(ag):
#    code = ag.agent_to_code("Cuauhtémoc Mayor Francisco Arcos")
#    assert code['code_1'] == "GOV"
#    assert code['country'] == "MEX"
        


def test_def_min_name(ag):
    code = ag.agent_to_code("Diego Molano", query_date="2022-06-01")
    assert code['country'] == "COL"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

def test_def_min_title(ag):
    code = ag.agent_to_code("Defense Minister")
    assert code['country'] == ""
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

@pytest.mark.skip(reason="Context currently can't be used to add country info")
def test_def_min_title_context(ag):
    code = ag.agent_to_code("Defense Minister", "German")
    assert code['country'] == "DEU"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

def test_def_min_full(ag):
    d = {"actor": "Diego Molano",
        "context": "Colombian defense",
        "date": "Today",
        "correct_country": "COL",
        "correct_code1": "GOV",
        "correct_code2": "MIL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_mongolia(ag):
    code = ag.agent_to_code("Mongolia", "", "today")
    assert code['country'] == "MNG"
    assert code['code_1'] == ""

def test_time1(ag):
    code = ag.agent_to_code("Angela Merkel", "", "2022-01-10")
    assert code['country'] == "DEU"
    assert code['code_1'] == "ELI"

def test_time2(ag):
    code = ag.agent_to_code("Angela Merkel", "", "2017-04-01")
    assert code['wiki'] == "Angela Merkel"
    assert code['actor_wiki_job'] == "Chancellor of Germany"
    # note: NOT "Leader of the Christian Democratic Union", which she
    # also was, but which the infobox (rightly) gives a lower priority
    assert code['country'] == "DEU"
    assert code['code_1'] == "GOV"

def test_drian_name_past(ag):
    code = ag.agent_to_code("Jean-Yves Le Drian", "", "10 May 2015")
    assert code['country'] == "FRA"
    assert code['code_1'] == "GOV"

def test_mongolian_head(ag):
    code = ag.agent_to_code("Ukhnaagiin Khurelsukh", "", "today")
    assert code['country'] == "MNG"
    assert code['code_1'] == "GOV"

def test_mongolian_head_umlauts(ag):
    code = ag.agent_to_code("Ukhnaagiin Khürelsükh")
    assert code['country'] == "MNG"
    assert code['code_1'] == "GOV" 

def test_mongolian_title(ag):   
    # the specific issue with this one is that the matched pattern
    # is "regional governor", not "president"
    code = ag.agent_to_code("the Mongolian President")
    assert code['country'] == "MNG"
    assert code['code_1'] == "GOV"

@pytest.mark.skip(reason="Address 'ruling party' question")
def test_mongolian_pp(ag):
    code = ag.agent_to_code("the ruling Mongolian People's Party", "", "October 1, 2021")
    assert code['code_1'] == "????"
    assert code['country'] == "MNG"

def test_chilean(ag):
    code = ag.agent_to_code("Minister Juan Carlos Jobet", "", "2022-04-01")
    assert code['country'] == "CHL"
    assert code['code_1'] == "GOV"

def test_chilean_2(ag):
    code = ag.agent_to_code("Juan Carlos Jobet", "", "2022-04-01")
    assert code['country'] == "CHL"
    assert code['code_1'] == "GOV"

def test_chile_mines(ag):
    code = ag.agent_to_code("Chile's Ministry of Mines")
    assert code['country'] == "CHL"
    assert code['code_1'] == "GOV"

def test_chile_defense_dept(ag):
    code = ag.agent_to_code("Chile's Defense Department")
    assert code['country'] == "CHL"
    assert code['code_1'] == "GOV"

def test_syr_mercs(ag):
    code = ag.agent_to_code("Syrian mercenaries")
    assert code['country'] == "SYR"
    assert code['code_1'] == "UAF"

def test_burkhard(ag):
    code = ag.agent_to_code("Thierry Burkhard")
    assert code['wiki'] == "Thierry Burkhard"
    assert code['country'] == "FRA"
    assert code['code_1'] == "MIL"

def test_general_burkhard(ag):
    code = ag.agent_to_code("General Thierry Burkhard")
    assert code['wiki'] == "Thierry Burkhard"
    assert code['country'] == "FRA"
    assert code['code_1'] == "MIL"

def test_chile_defense_min(ag):
    code = ag.agent_to_code("Chile's Defense Ministry")
    assert code['country'] == "CHL"
    assert code['code_1'] == "GOV"


def senate_hopeful(ag):
    code = ag.agent_to_code("Senate hopeful")
    assert code['country'] == ""
    assert code['code_1'] == "PTY"

def test_denmark_union(ag):
    code = ag.agent_to_code("Several of the largest trade unions in Denmark")
    assert code['country'] == "DNK"
    assert code['code_1'] == "LAB"

def test_100(ag):
    code = ag.agent_to_code("Pakistani Army")
    assert code['country'] == "PAK"
    assert code['code_1'] == "MIL"   

def test_101(ag):
    code = ag.agent_to_code("Saudi National Guard")
    assert code['country'] == "SAU"
    assert code['code_1'] == "MIL"  

def test_102(ag):
    code = ag.agent_to_code("Saudi Arabia")
    assert code['country'] == "SAU"
    assert code['code_1'] == ""  

def test_103(ag):
    code = ag.agent_to_code("A national intelligence agency officer")
    assert code['country'] == ""
    assert code['code_1'] == "SPY"  

def test_104(ag):
    # moving NGO from code_1 to top level/country happens elsewhere
    code = ag.agent_to_code("Red Crescent Society")
    assert code['country'] == ""
    assert code['code_1'] == "NGO"  
    assert code['wiki'] == "International Red Cross and Red Crescent Movement"  

def test_105(ag):
    code = ag.agent_to_code("Boyko Borisov", query_date="12 April 2022")
    assert code['country'] == "BGR"
    assert code['code_1'] == "GOV"  
    assert code['wiki'] == "Boyko Borisov"  

def test_106(ag):
    code = ag.agent_to_code("United Nations Security Council")
    assert code['country'] == "UNO"

def test_107(ag):
    code = ag.agent_to_code("A team of heavily armed police")
    assert code['country'] == ""
    assert code['code_1'] == "COP"  
    assert code['wiki'] == ""  

def test_108(ag):
    code = ag.agent_to_code("Minsk police station")
    assert code['country'] == "BLR"
    assert code['code_1'] == "COP"  

def test_109(ag):
    # currently failing because the name alone isn't being recognized as an entity
    code = ag.agent_to_code("Agathon Rwasa")
    assert code['wiki'] == "Agathon Rwasa"  
    assert code['country'] == "BDI"
    assert code['code_1'] == "GOV"  

def test_110(ag):
    code = ag.agent_to_code("Serbian Army")
    assert code['country'] == "SRB"
    assert code['code_1'] == "MIL"  

def test_111(ag):
    code = ag.agent_to_code("South Africa")
    assert code['country'] == "ZAF"
    assert code['code_1'] == ""  
    assert code['wiki'] == ""  

def test_112(ag):
    code = ag.agent_to_code("Vladimir Putin", "", "today")
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"  
    assert code['wiki'] == "Vladimir Putin"  

def test_113(ag):
    code = ag.agent_to_code("Egypt")
    assert code['country'] == "EGY"
    assert code['source'] == "country only"  
    
def test_114(ag):
    code = ag.agent_to_code("Chief Minister Jam Kamal Khan Alyani", "", "2022-10-20")
    assert code['wiki'] == "Jam Kamal Khan"  
    assert code['country'] == "PAK"
    assert code['code_1'] == "LEG"  

def test_114_2(ag):
    # TODO: Here's a place where we might want to change the similarity calculation.
    # Maybe something like edit distance of words? Okay to delete a middle name or
    # add another name on the end?
    code = ag.agent_to_code("Jam Kamal Khan Alyani",  "", "2022-10-20")
    assert code['wiki'] == "Jam Kamal Khan"  
    assert code['country'] == "PAK"
    assert code['code_1'] == "LEG"  

def test_114_3(ag):
    code = ag.agent_to_code("Jam Kamal Khan", "", "2022-10-20")
    assert code['wiki'] == "Jam Kamal Khan"  
    assert code['country'] == "PAK"
    assert code['code_1'] == "LEG"  

def test_114_4(ag):
    # Should prioritize first office that matches the date range
    code = ag.agent_to_code("Jam Kamal Khan", "", "2021-10-01")
    assert code['wiki'] == "Jam Kamal Khan"  
    assert code['country'] == "PAK"
    assert code['code_1'] == "GOV"  

def test_115(ag):
    code = ag.agent_to_code("Marcus Beam")
    assert code is None 

def test_116(ag):
    code = ag.agent_to_code("regions across Indonesia")
    assert code['country'] == "IDN"
    assert code['code_1'] == "UNK"  

#def test_117(ag):
#    code = ag.agent_to_code("Soy Sopheap")
#    assert code['country'] == "KHM"
#    assert code['code_1'] == "GOV"  
#    assert code['wiki'] == ""  

def test_118(ag):
    # MISSING WIKI
    code = ag.agent_to_code("Evelyne Anite")
    assert code['wiki'] == "Evelyn Anite"  
    assert code['country'] == "UGA"
    assert code['code_1'] == "PTY"  

def test_119(ag):
    code = ag.agent_to_code("Republika Srpska")
    assert code['country'] == "BIH"
    assert code['wiki'] == "Republika Srpska"  

def test_120(ag):
    code = ag.agent_to_code("Slovak Soldiers")
    assert code['country'] == "SVK"
    assert code['code_1'] == "MIL"  
    assert code['wiki'] == ""  

def test_121(ag):
    code = ag.agent_to_code("President Ibrahim Mohamed Solih", query_date="2022-05-01")
    assert code['wiki'] == "Ibrahim Mohamed Solih"  
    assert code['country'] == "MDV"
    assert code['code_1'] == "GOV"  

def test_121_2(ag):
    code = ag.agent_to_code("Ibrahim Mohamed Solih", query_date="2022-05-01")
    assert code['wiki'] == "Ibrahim Mohamed Solih"  
    assert code['country'] == "MDV"
    assert code['code_1'] == "GOV"  

def test_122(ag):
    code = ag.agent_to_code("paramilitary organization")
    assert code['country'] == ""
    assert code['code_1'] == "PRM"  

def test_123(ag):
    code = ag.agent_to_code("US Senator Marco Rubio", query_date="2022-03-01")
    assert code['wiki'] == "Marco Rubio"  
    assert code['country'] == "USA"
    assert code['code_1'] == "LEG"  

def test_124(ag):
    # random dude, shouldn't get coded as anything
    # Problem with the fuzzy search: he's now getting matched
    # to some guy named "Joseph M. Monks"
    code = ag.agent_to_code("Joseph Monka")
    assert code is None

def test_125(ag):
    code = ag.agent_to_code("Norway's central bank")
    assert code['country'] == "NOR"
    assert code['code_1'] == "GOV"  
    assert code['wiki'] == ""  

def test_126(ag):
    code = ag.agent_to_code("two Chinese warplanes")
    assert code['country'] == "CHN"
    assert code['code_1'] == "MIL"  
    assert code['wiki'] == ""  

def test_127(ag):
    code = ag.agent_to_code("United States")
    assert code['country'] == "USA"
    assert code['code_1'] == ""  
    assert code['wiki'] == ""  

def test_128(ag):
    code = ag.agent_to_code("Ukrainian agricultural workers")
    assert code['country'] == "UKR"
    assert code['code_1'] == "AGR"  
    assert code['wiki'] == ""  

def test_128_2(ag):
    code = ag.agent_to_code("Ukrainian farmers")
    assert code['country'] == "UKR"
    assert code['code_1'] == "AGR"  
    assert code['wiki'] == ""  

def test_129(ag):
    code = ag.agent_to_code("Alexander Lukashenko")
    assert code['country'] == "BLR"
    assert code['code_1'] == "GOV"  
    assert code['wiki'] == "Alexander Lukashenko"  

def test_130(ag):
    code = ag.agent_to_code("A Belarusian protester")
    assert code['country'] == "BLR"
    assert code['code_1'] == "CVL"  
    assert code['code_2'] == "OPP"  

def test_131(ag):
    code = ag.agent_to_code("chairman of the company's board")
    assert code['country'] == ""
    assert code['code_1'] == "BUS"  
    assert code['wiki'] == ""  

def test_132(ag):
    code = ag.agent_to_code("a group of tourists")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"  
    assert code['pattern'] == "tourist"  


def test_date_putin1(ag):
    code = ag.agent_to_code("Vladimir Putin", query_date="2022-03-01")
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"  
    assert code['actor_wiki_job'] == "President of Russia"  

def test_date_putin2(ag):
    code = ag.agent_to_code("Vladimir Putin", query_date="2009-01-01")
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"  
    assert code['actor_wiki_job'] == "Prime Minister of Russia"  

def test_date_putin3(ag):
    # NOTE: this correctly gets the title "director of the federal security service".
    # I've added this manually to the agents file, but that's not very sustainable.
    # A better but very annoying solution is to recursively do a Wiki lookup on the
    # titles themselves. 😑
    code = ag.agent_to_code("Vladimir Putin", query_date="1999-01-01")
    assert code['actor_wiki_job'] == "Director of the Federal Security Service"  
    assert code['wiki'] == "Vladimir Putin"
    assert code['country'] == "RUS"
    assert code['code_1'] == "SPY"  

@pytest.mark.xfail(reason="I don't see an easy way to fix this. Jeep --> USA BUS")
def test_jeep(ag):
    code = ag.agent_to_code("Jeep driver")
    assert code['country'] == ""

def test_somali_pm(ag):
    d = {"actor": "Somalia's prime minister",
        "context": "",
        "date": "today",
        "correct_country": "SOM",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        
def test_roble(ag):
    ## manually add alternative Mohammed spellings?
    # Otherwise, don't require exact matches and develop a better
    # wiki scoring system.
    code = ag.agent_to_code("Mohammed Hussein Roble", "", "2022-03-01")
    assert code['wiki'] == "Mohamed Hussein Roble"
    assert code['country'] == "SOM"
    assert code['code_1'] == "GOV"
        
@pytest.mark.xfail(reason="SONNA doesn't have a Wikipedia page") 
def test_somali_state_media(ag):
    code = ag.agent_to_code("SONNA", "Somalia")
    assert code['country'] == "SOM"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "JRN"
        
def test_somali_state_media2(ag):
    code = ag.agent_to_code("Somalia state news agency")
    assert code['country'] == "SOM"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "JRN"
        
        
def test_macron_full(ag):
    d = {"actor": "French President Emmanuel Macron ",
        "context": "",
        "date": "Today",
        "correct_country": "FRA",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        

def test_macron_name(ag):
    code = ag.agent_to_code("Emmanuel Macron", "", "2022-03-01")
    assert code['wiki'] == "Emmanuel Macron"
    assert code['country'] == "FRA"
    assert code['code_1'] == "GOV"
        

def test_french_pres(ag):
    d = {"actor": "French President",
        "context": "",
        "date": "Today",
        "correct_country": "FRA",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        
        
def test_drian_full(ag):
    code = ag.agent_to_code("French Foreign Minister Jean-Yves Le Drian", query_date="2022-03-01")
    assert code['wiki'] == 'Jean-Yves Le Drian'
    assert code['code_1'] == "GOV"
    assert code['country'] == "FRA"
        

def test_drian_name(ag):
    code = ag.agent_to_code("Jean-Yves Le Drian", query_date="2022-03-01")
    assert code['code_1'] == "GOV"
    assert code['country'] == "FRA"
    assert code['wiki'] == 'Jean-Yves Le Drian'
        
        
def test_tseng_full(ag):
    d = {"actor": "Harry Ho-jen Tseng, Taiwan’s deputy foreign minister",
        "context": "",
        "date": "Today",
        "correct_country": "TWN",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        

def test_tseng_name(ag):
    code = ag.agent_to_code("Harry Ho-jen Tseng", "", "2022-01-01")
    assert code['wiki'] == "Tseng Hou-jen"
    assert code['country'] == "TWN"
    assert code['code_1'] == "GOV"
        

def test_tseng_name2(ag):
    d = {"actor": "Tseng Ho-Jen",
        "context": "",
        "date": "Today",
        "correct_country": "TWN",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code("Tseng Hou-jen", "", "2022-01-01")
    assert code['wiki'] == "Tseng Hou-jen"
    assert code['country'] == "TWN"
    assert code['code_1'] == "GOV"
        

def test_lith_full(ag):
    d = {"actor": "Lithuania’s president Gitanas Nauseda",
        "context": "",
        "date": "Today",
        "correct_country": "LTU",
        "correct_code1": "GOV",
        "wiki": "Gitanas Nausėda"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    assert code['wiki'] == d['wiki']
        

def test_lith_name(ag):
    d = {"actor": "Gitanas Nauseda",
        "context": "",
        "date": "Today",
        "correct_country": "LTU",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

@pytest.mark.xfail(reason="Can't handle people's titles if they're not in Wiki alt names")
def test_col_sen(ag):
    d = {"actor": "Sen. Feliciano Valencia",
        "context": "",
        "date": "Today",
        "correct_country": "COL",
        "correct_code1": "LEG"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

@pytest.mark.xfail(reason="Not in Wikipedia")
def test_col_sen_name(ag):
    code = ag.agent_to_code("Feliciano Valencia")
    assert code['code_1'] == "COL"
    assert code['country'] == "LEG"
        
def test_national_guard_solo(ag):
    # TODO: If (1) there's no country in the text, AND
    #          (2) there's no context, AND
    #          (3) there's no person/org in the text, AND 
    #          (4) there's a near-perfect agent match:
    #           THEN: Don't look up on Wiki at all.
    code = ag.agent_to_code("the National Guard", "", "2022-01-11")
    assert code['country'] == ""
    assert code['code_1'] == "MIL"
        
@pytest.mark.skip(reason="Wiki searches currently don't use context")
def test_national_guard_context1(ag):
    code = ag.agent_to_code("the Mexican National Guard")
    assert code['country'] == "MEX"
    assert code['code_1'] == "MIL"
        
@pytest.mark.skip(reason="Wiki searches currently don't use context")
def test_national_guard_context2(ag):
    d = {"actor": "the National Guard",
        "context": "Saudi",
        "date": "Today",
        "correct_country": "SAU",
        "correct_code1": "MIL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_national_guard_saudi(ag):
    d = {"actor": "the Saudi National Guard",
        "context": "",
        "date": "Today",
        "correct_country": "SAU",
        "correct_code1": "MIL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country'] 

@pytest.mark.skip(reason="Not in Wikipedia")
def test_searchers(ag):
    # Mexican civilian volunteers/NGO/Human Rights-type group
    d = {"actor": "Searchers of Puerto Penasco",
        "context": "",
        "date": "Today",
        "correct_country": "",  # No wiki page, so leave blank
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_puerto(ag):
    # cities don't get a code 1
    wiki = ag.query_wiki("Puerto Penasco")
    assert wiki['title'] == 'Puerto Peñasco'
    code = ag.agent_to_code("Puerto Penasco")
    assert code['country'] == "MEX"
    assert code['code_1'] == ""

def test_FDLR_1(ag):
    code = ag.agent_to_code("the rebel FDLR force", "", "2022-01-01")
    assert code['wiki'] == 'Democratic Forces for the Liberation of Rwanda'
    assert code['country'] == "RWA" # Or DRC?
    assert code['code_1'] == "REB"
        
def test_FDLR_2(ag):
    # TODO: The wiki page is missing a short description, so it's using the "war faction"
    # infobox title, which isn't getting coded as rebel. The first sentence describes it
    # as an armed rebel group in the DRC, which might be an argument for using the first
    # sentence again.
    code = ag.agent_to_code("FDLR", "", "2022-01-01")
    assert code['wiki'] == 'Democratic Forces for the Liberation of Rwanda'
    assert code['country'] == "RWA" # Or DRC?
    assert code['code_1'] == "REB"
                
def test_SDF_1(ag):
    code = ag.agent_to_code("Kurdish-led Syrian Democratic Forces", "", "2022-01-01")
    assert code['country'] == "SYR"
    assert code['code_1'] == "REB"
        
def test_SDF_2(ag):
    # TODO: Another example where the simple agent lookup fails and it should
    # do a Wikipedia query instead
    code = ag.agent_to_code("Syrian Democratic Forces", "", "2022-01-01")
    assert code['country'] == "SYR"
    assert code['code_1'] == "REB"
        
def test_taxi(ag):      
    code = ag.agent_to_code("taxi drivers")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"
        
def test_eu(ag): 
    code = ag.agent_to_code("the European Union")
    assert code['country'] == "EUR"
        
def test_eu_leg1(ag): 
    code = ag.agent_to_code("European Union legislators")
    assert code['country'] == "EUR"
    assert code['code_1'] == "LEG"
        
def test_eu_leg2(ag): 
    code = ag.agent_to_code("the EU legislature")
    assert code['country'] == "EUR"
    assert code['code_1'] == "LEG"
               
def test_rak_reb(ag):
    code = ag.agent_to_code("Rakhine rebels")
    assert code['country'] == "MMR"
    assert code['code_1'] == "REB"
        
def test_rak_bud(ag):
    code = ag.agent_to_code("Rakhine Buddhists")
    assert code['country'] == "MMR"
    assert code['code_1'] == "BUD"

def test_rak_reb(ag):
    code = ag.agent_to_code("Rakhine rebels")
    assert code['country'] == "MMR"
    assert code['code_1'] == "REB"

@pytest.mark.xfail(reason="Wiki doesn't have his old job in the info box")
def test_kartapolov_full(ag):
    code = ag.agent_to_code("Deputy Minister of Defense Colonel-General Andrei Kartapolov", "", "2021-01-01")
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

@pytest.mark.xfail(reason="Wiki doesn't have his old job in the info box")
def test_kartapolov_name(ag):
    code = ag.agent_to_code("Andrei Kartapolov", "", "2021-01-01")
    assert code['wiki'] == "Andrey Kartapolov"
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"
            
def spokesman(ag):
    d = {"actor": "The provincial governor's spokesman, Mohammad Arif Nuri",
        "context": "",
        "date": "Today",
        "correct_country": "",  # from context, AFG, but no wiki
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
                
def test_jewish(ag):
    code = ag.agent_to_code("A prominent Jewish community leader")
    assert code['country'] == ""
    assert code['code_1'] == "REL"
                
def test_ecu_ombud(ag):
    d = {"actor": "Ecuador's Ombudsman's Office",
        "context": "",
        "date": "Today",
        "correct_country": "ECU",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code("Ecuador's Ombudsman's Office", "", "today")
    assert code['code_1'] == "GOV"
    assert code['country'] == "ECU"
                
def test_indig(ag):
    d = {"actor": "an indigenous leader",
        "context": "",
        "date": "Today",
        "correct_country": "",
        "correct_code1": "CVL"  # maybe?? or GOV?
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_boko_haram(ag):
    code = ag.agent_to_code("Boko Haram")
    assert code['wiki'] == "Boko Haram"
    assert code['country'] == "NGA"
    assert code['code_1'] == "REB"

def test_boko_haram2(ag):
    code = ag.agent_to_code("Boko Haram gunmen")
    assert code['country'] == "NGA"
    assert code['code_1'] == "REB"
        
def test_som_reb(ag):
    # TODO: wikipedia describes this a paramilitary group.
    # The categories have it listed as a rebel group. Maybe rely more
    # on the categories? There's just so much junk in there, though...
    code = ag.agent_to_code("Ahlu Sunnah Wa-Jama", "", "2022-02-01")
    assert code['wiki'] == "Ahlu Sunna Waljama'a"
    assert code['country'] == "SOM"
    assert code['code_1'] == "REB"

def test_som_reb2(ag):
    code = ag.agent_to_code("Ahlu Sunnah Wa-Jama", "Somalia", "2022-02-01")
    assert code['wiki'] == "Ahlu Sunna Waljama'a"
    assert code['country'] == "SOM"
    assert code['code_1'] == "PRM"

def test_som_reb3(ag):
    code = ag.agent_to_code("Ahlu Sunnah Waljama", "Somalia", "2022-02-01")
    assert code['wiki'] == "Ahlu Sunna Waljama'a"
    assert code['country'] == "SOM"
    assert code['code_1'] == "PRM"

def test_som_reb4(ag):
    code = ag.agent_to_code("a Somalia-based paramilitary group")
    assert code['country'] == "SOM"
    assert code['code_1'] == "PRM"

def test_polit_pris(ag):  
    code = ag.agent_to_code("political prisoners")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"
    assert code['code_2'] == "OPP"
                
def test_nawaf(ag):
    d = {"actor": "Nawaf Al-Ahmad Al-Jaber Al-Sabah",
        "context": "",
        "date": "Today",
        "correct_country": "KWT",
        "correct_code1": "GOV",
        "correct_wiki": "Nawaf Al-Ahmad Al-Jaber Al-Sabah"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    assert code['wiki'] == d['correct_wiki']
        
#def test_nawaf_past(ag):
#    d = {"actor": "Nawaf Al-Ahmad Al-Jaber Al-Sabah",
#        "context": "",
#        "date": "1 January 2018",
#        "correct_country": "KWT",
#        "correct_code1": "GOV"  # He was still part of the gov, just not ruler
#        }


def test_ratas_leg(ag):
    code = ag.agent_to_code("Jüri Ratas", "", "1 july 2021")
    assert code['country'] == "EST"
    assert code['code_1'] == "LEG"

def test_ratas_gov(ag):
    d = {"actor": "Jüri Ratas",
        "context": "",
        "date": "1 January 2018",
        "correct_country": "EST",
        "correct_code1": "GOV" # prime minister
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    
def test_ratas_office(ag):
    d = {"actor": "18th Prime Minister of Estonia",
        "context": "",
        "date": "today",
        "correct_country": "EST",
        "correct_code1": "GOV" # prime minister
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
        
def test_president(ag):
    code = ag.agent_to_code("president")
    assert code['country'] == ""
    assert code['code_1'] == "GOV"
    assert code['wiki'] == ''

def test_riigikogu(ag):
    code = ag.agent_to_code("Riigikogu")
    assert code['wiki'] == 'Riigikogu'
    assert code['country'] == "EST"
    assert code['code_1'] == "LEG"

def test_pres_riigikogu(ag):
    # Another case where we don't want to stop with the 
    # inital lookup and want to make sure to query Wiki.
    code = ag.agent_to_code("president of the Riigikogu")
    assert code['country'] == "EST"
    assert code['code_1'] == "LEG"
    
        
def test_fsb(ag):
    code = ag.agent_to_code("FSB")
    assert code['country'] == "RUS"
    assert code['code_1'] == "SPY"

def test_fsb(ag):
    code = ag.agent_to_code("FSB", "Russia")
    assert code['country'] == "RUS"
    assert code['code_1'] == "SPY" 

def test_fsb2(ag):
    d = {"actor": "FSB, Russia’s security service",
        "context": "",
        "date": "Today",
        "correct_country": "RUS",
        "correct_code1": "SPY"  # ??? Or COP?
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    

def test_gru(ag):
    code = ag.agent_to_code("GRU")
    assert code['wiki'] == "GRU"
    assert code['country'] == "RUS"
    assert code['code_1'] == "MIL"
    assert code['code_1'] == "SPY" 

def mil_spy(ag):
    code = ag.agent_to_code("foreign military intelligence agency")
    assert code['country'] == ""
    assert code['code_1'] == "MIL" 
    assert code['code_1'] == "SPY" 

def test_beijing(ag):
    code = ag.agent_to_code("Beijing")
    assert code['country'] == "CHN"
    assert code['code_1'] == "GOV"
    
        
def test_fra_mus(ag):
    code = ag.agent_to_code("Muslim leaders in France")
    assert code['country'] == "FRA"
    assert code['code_1'] == "REL"
    

def test_icj(ag):
    code = ag.agent_to_code("International Court of Justice", "", "today")
    assert code['country'] == "UNO"
    assert code['code_1'] == "JUD"
        
def test_demonym1(ag):
    d = {"actor": "Palestinians",
        "context": "",
        "date": "Today",
        "correct_country": "PSE",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    

def test_demonym2(ag):
    d = {"actor": "Brazilians",
        "context": "",
        "date": "Today",
        "correct_country": "BRA",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']


def test_demonym3(ag):
    d = {"actor": "British",
        "context": "",
        "date": "Today",
        "correct_country": "GBR",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    
    

def test_country1(ag):
    d = {"actor": "UK",
        "context": "",
        "date": "Today",
        "correct_country": "GBR",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    

def test_country2(ag):
    d = {"actor": "Great Britain",
        "context": "",
        "date": "Today",
        "correct_country": "GBR",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    
def test_country3(ag):
    d = {"actor": "Japan",
        "context": "",
        "date": "Today",
        "correct_country": "JPN",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_country4(ag):
    d = {"actor": "France",
        "context": "",
        "date": "Today",
        "correct_country": "FRA",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_palestinian_auth(ag):
    d = {"actor": "Palestinian Authority",
        "context": "",
        "date": "Today",
        "correct_country": "PSE",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def tony_blair_now(ag):
    d = {"actor": "Tony Blair",
        "context": "",
        "date": "1 January 2003",
        "correct_country": "GBR",
        "correct_code1": "GOV"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country'] 


def tony_blair_2003(ag):
    d = {"actor": "Tony Blair",
        "context": "",
        "date": "Today",
        "correct_country": "GBR",
        "correct_code1": "ELI"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country'] 
        
def test_kosovo(ag):
    d = {"actor": "Kosovo",
        "context": "",
        "date": "Today",
        "correct_country": "XKX",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']
    

def test_taiwan(ag):
    code = ag.agent_to_code("Taiwan")
    assert code['country'] == "TWN"
    assert code['code_1'] == ""

def test_pre01(ag):
    code = ag.agent_to_code("partially recognized country")
    assert code['country'] == ""
    assert code['code_1'] == "PRE"

def test_pre02(ag):
    code = ag.agent_to_code("breakaway state")
    assert code['country'] == ""
    assert code['code_1'] == "PRE"

def test_pre1(ag):
    code = ag.agent_to_code("Abkhazia")
    assert code['country'] == "GEO"
    assert code['code_1'] == "PRE"

def test_pre2(ag): 
    code = ag.agent_to_code("South Ossetia")
    assert code['country'] == "GEO"
    assert code['code_1'] == "PRE"

def test_pre3(ag):   
    code = ag.agent_to_code("Transnistria")
    assert code['country'] == "MDA"
    assert code['code_1'] == "PRE"

def test_pre4(ag):   
    code = ag.agent_to_code("Nagorno-Karabakh")
    assert code['country'] == "AZE"
    assert code['code_1'] == "PRE"

def test_pre5(ag):   
    code = ag.agent_to_code("Somaliland")
    assert code['country'] == "SOM"
    assert code['code_1'] == "PRE"

def test_pre6(ag):   
    code = ag.agent_to_code("Northern Cyprus")
    assert code['country'] == "CYP"
    assert code['code_1'] == "PRE"


def test_gunmen(ag):
    code = ag.agent_to_code("gunmen")
    assert code['country'] == ""
    assert code['code_1'] == "UAF"
    
        
def test_quds(ag):
    code = ag.agent_to_code("Quds Force")
    assert code['wiki'] == "Quds Force"
    assert code['country'] == "IRN"
    assert code['code_1'] == "MIL"

def junk(ag):    
    # Do we want to return None, [], or a full dict with empty values?
    d = {"actor": "aljsndgojnown",
        "context": "",
        "date": "Today",
        "correct_country": "",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert not code

def dozen_men(ag):        
    code = ag.agent_to_code("over two dozen men and women")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"
        
def test_junk2(ag):
    d = {"actor": "a demand",
        "context": "",
        "date": "Today",
        "correct_country": "",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code == None

def test_targets_in_syria(ag):
    d = {"actor": "targets in Syria",
        "context": "",
        "date": "Today",
        "correct_country": "SYR",
        "correct_code1": ""
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']


def test_mus_cleric(ag):  
    # Currently failing because it queries Wiki for "muslim" and gets 
    # a (weird) Wiki article. If the confidence is high on the text match,
    # just skip Wiki altogether??
    d = {"actor": "Muslim cleric",
        "context": "",
        "date": "Today",
        "correct_country": "",
        "correct_code1": "REL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']


def am_soldiers(ag): 
    d = {"actor": "American soldiers",
        "context": "",
        "date": "Today",
        "correct_country": "USA",
        "correct_code1": "MIL"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def ukr_farmers(ag):
    d = {"actor": "Ukrainian farmers",
        "context": "",
        "date": "Today",
        "correct_country": "UKR",
        "correct_code1": "AGR"
        }
    code = ag.agent_to_code(d['actor'], d['context'], d['date'])
    assert code['code_1'] == d['correct_code1']
    assert code['country'] == d['correct_country']

def test_201(ag): 
    # TODO: handle country info better.
    code = ag.agent_to_code("Canadian Ambassador to the United Nations")
    assert code['wiki'] == 'Permanent Representative of Canada to the United Nations'
    assert code['country'] == "CAN"
    assert code['code_1'] == "GOV"

def test_202(ag): 
    code = ag.agent_to_code("a disgraced former minister")
    assert code['country'] == ""       
    assert code['code_1'] == "ELI"

def test_203(ag): 
    code = ag.agent_to_code("puppies")
    assert code is None

def test_204(ag): 
    # Matching the agents file, not querying Wiki
    code = ag.agent_to_code("Central Intelligence Agency")
    assert code['wiki'] == "Central Intelligence Agency" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "SPY" 

def test_205(ag): 
    code = ag.agent_to_code("CIA")
    assert code['wiki'] == "Central Intelligence Agency" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "SPY" 

def test_206(ag): 
    code = ag.agent_to_code("a former CIA official")
    assert code['country'] == "USA"       
    assert code['code_1'] == "ELI" 

def test_207(ag): 
    code = ag.agent_to_code("a former White House official")
    assert code['country'] == "USA"       
    assert code['code_1'] == "ELI" 

def test_208(ag): 
    code = ag.agent_to_code("official residence and workplace of the President of the United States")
    assert code['country'] == "USA"       
    assert code['code_1'] == "GOV" 

def test_208_2(ag): 
    code = ag.agent_to_code("coronavirus")
    assert code is None

def test_209(ag): 
    code = ag.agent_to_code("Russia's Baltic Fleet")
    assert code['country'] == "RUS"       
    assert code['code_1'] == "MIL" 
        
def test_210(ag): 
    code = ag.agent_to_code("the Baltic Fleet")
    assert code['wiki'] == "Baltic Fleet" 
    assert code['country'] == "RUS"       
    assert code['code_1'] == "MIL" 

def test_210(ag): 
    code = ag.agent_to_code("an Arleigh Burke-class destroyer")
    assert code['country'] == "USA"       
    assert code['code_1'] == "MIL" 

def test_210(ag): 
    code = ag.agent_to_code("Beyonce")
    assert code['wiki'] == "Beyoncé" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "CVL" 

def test_211(ag): 
    code = ag.agent_to_code("Massachusetts Institute of Technology")
    assert code['wiki'] == "Massachusetts Institute of Technology" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "EDU" 

def test_212(ag): 
    code = ag.agent_to_code("Harvard")
    assert code['wiki'] == "Harvard University"   
    assert code['country'] == "USA"       
    assert code['code_1'] == "EDU" 

def test_213(ag): 
    # Another one where it's just hitting the agent file,
    # not querying Wiki like it should.
    code = ag.agent_to_code("Oxford University")
    assert code['wiki'] == "University of Oxford"   
    assert code['country'] == "GBR"       
    assert code['code_1'] == "EDU" 

def test_214(ag): 
    code = ag.agent_to_code("Amherst College")
    assert code['wiki'] == "Amherst College"
    assert code['country'] == "USA"
    assert code['code_1'] == "EDU"

def test_215(ag): 
    code = ag.agent_to_code("Beijing")
    assert code['wiki'] == "Beijing" 
    assert code['country'] == "CHN"       
 
def test_216(ag): 
    code = ag.agent_to_code("Kathleen Hicks", query_date="2022-04-01")
    assert code['wiki'] == "Kathleen Hicks" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "GOV" 
    assert code['code_2'] == "MIL" 

def test_216_2(ag): 
    code = ag.agent_to_code("Deputy Secretary Hicks", query_date="2022-04-01")
    assert code['wiki'] == "Kathleen Hicks" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "GOV" 
    assert code['code_2'] == "MIL" 

def test_217(ag): 
    code = ag.agent_to_code("DAAD")
    assert code['wiki'] == "German Academic Exchange Service" 
    assert code['country'] == ""       
    #assert code['code_1'] == "GOV"  

def test_218(ag): 
    code = ag.agent_to_code("DfID")
    assert code['wiki'] == "Department for International Development" 
    assert code['country'] == "GBR"       
    assert code['code_1'] == "GOV"  
        
def test_219(ag): 
    code = ag.agent_to_code("DfID")
    assert code['wiki'] == "Department for International Development" 
    assert code['country'] == "GBR"       
    assert code['code_1'] == "GOV"  
     
def test_220(ag): 
    code = ag.agent_to_code("Hashim Thaci", "", "2011-12-01")
    assert code['wiki'] == "Hashim Thaçi" 
    assert code['country'] == "XKX"       
    assert code['code_1'] == "GOV"  

def test_220_2(ag): 
    code = ag.agent_to_code("Hashim Thaci", "", "today")
    assert code['wiki'] == "Hashim Thaçi" 
    assert code['country'] == "XKX"       
    assert code['code_1'] == "ELI"  

def test_221(ag): 
    code = ag.agent_to_code("Albin Kurti", "", "2021-01-01")
    assert code['wiki'] == "Albin Kurti" 
    assert code['country'] == "XKX"       
    assert code['code_1'] == "PTY"        
    assert code['code_2'] == "OPP"

def test_221_2(ag): 
    code = ag.agent_to_code("Albin Kurti", "", "2021-10-01")
    assert code['wiki'] == "Albin Kurti" 
    assert code['country'] == "XKX"       
    assert code['code_1'] == "GOV"     

def test_223(ag): 
    code = ag.agent_to_code("Prishtina")
    assert code['wiki'] == "Pristina" 
    assert code['country'] == "XKX"       

#def test_224(ag): 
#    code = ag.agent_to_code("Tartar experts")
#    assert code['country'] == "RUS" 
#    assert code['code_1'] == "CIV"       

def test_225(ag): 
    # TODO: Fix date issue
    code = ag.agent_to_code("Mahinda Rajapaksa", query_date="2022-05-01")
    assert code['wiki'] == "Mahinda Rajapaksa" 
    assert code['country'] == "LKA"   
    assert code['code_1'] == "GOV"   
    assert code['actor_wiki_job'] == "Prime Minister of Sri Lanka"

def test_226(ag): 
    code = ag.agent_to_code("Gotabaya Rajapaksa", query_date="2022-05-01")
    assert code['wiki'] == "Gotabaya Rajapaksa" 
    assert code['country'] == "LKA"   
    assert code['code_1'] == "GOV"   
    assert code['actor_wiki_job'] == "8th President of Sri Lanka"

def test_227(ag): 
    code = ag.agent_to_code("an opposition lawmaker")
    assert code['country'] == ""   
    assert code['code_1'] == "LEG"   

def test_228(ag): 
    # a non-famous person mentioned once in a NYT article
    code = ag.agent_to_code("Zulhijjah Mirzadah")
    assert code is None

def test_229(ag): 
    # TODO: prefer Wiki over everything else
    code = ag.agent_to_code("Taliban", query_date="2022-05-01")
    assert code['country'] == "AFG"
    assert code['code_1'] == "GOV"

def test_230(ag): 
    ## Problem with coding of historical mentions
    # Specifically, the box type is the current role and there's
    # not an easy way to get its past role.
    code = ag.agent_to_code("Taliban", query_date="2002-05-01")
    assert code['country'] == "AFG"
    assert code['code_1'] == "REB"

def test_231(ag): 
    code = ag.agent_to_code("an amusement park in Kabul")
    assert code['country'] == "AFG"
    assert code['code_1'] == "CVL"

def test_232(ag): 
    # settlements = CVL. Do we actually want that?
    code = ag.agent_to_code("Mariupol")
    assert code['country'] == "UKR"
    assert code['code_1'] == "CVL"

def test_233(ag): 
    code = ag.agent_to_code("professor of strategic studies at the University of St. Andrews in Scotland")
    assert code['country'] == "GBR"
    assert code['code_1'] == "EDU"

def test_234(ag): 
    code = ag.agent_to_code("antiwar protesters")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"
    assert code['code_1'] == "OPP"

def test_235(ag): 
    code = ag.agent_to_code("Britain’s Defense Ministry")
    assert code['country'] == "GBR"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

def test_236(ag): 
    code = ag.agent_to_code("president of the European Council")
    assert code['country'] == "EUR"
    assert code['code_1'] == "GOV"

def test_city_actors_1(ag): 
    # Need to separate out agents and proper names
    # OR, hacky thing: allow the rest of the phrase to overrule CVL when CVL comes 
    # from a Wikipedia settlement tag? That's a real nightmare though...
    code = ag.agent_to_code("Shanghai authorities")
    assert code['country'] == "CHN"
    assert code['code_1'] == "GOV"

def test_city_actors_2(ag): 
    # Need to separate out agents and proper names
    code = ag.agent_to_code("Kyiv police")
    assert code['country'] == "UKR"
    assert code['code_1'] == "COP"

def test_238(ag): 
    code = ag.agent_to_code("authorities")
    assert code['country'] == ""
    assert code['code_1'] == "GOV"

def test_239(ag): 
    # another example of when it should check Wiki and not
    # stop with the agent match
    # MISSING WIKI
    code = ag.agent_to_code("Pope Francis")
    assert code['country'] == ""
    assert code['code_1'] == "REL"
    assert code['wiki'] == "Pope Francis"

def test_240(ag): 
    code = ag.agent_to_code("the official National News Agency")
    assert code['country'] == ""
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "JRN"

def test_241(ag): 
    code = ag.agent_to_code("the Lebanese president's office")
    assert code['country'] == "LBN"
    assert code['code_1'] == "GOV"

def test_242(ag): 
    # TODO: another issue where the short description is missing so it's using the infobox,
    # which is "war faction". Again, maybe look at the first sentence?
    # "CODECO is a loose association of various Lendu militia groups operating within the Democratic Republic of the Congo."
    code = ag.agent_to_code("CODECO")
    assert code['wiki'] == "CODECO"
    assert code['country'] == "COD"
    assert code['code_1'] == "PRM"

def test_243(ag): 
    # Missing country. 
    code = ag.agent_to_code("Mongwalu")
    assert code['wiki'] == "Mongbwalu"
    assert code['country'] == "COD"
    assert code['code_1'] == "CVL"

def test_244(ag): 
    code = ag.agent_to_code("DRC")
    assert code['country'] == "COD"
    assert code['code_1'] == ""

def test_245(ag): 
    code = ag.agent_to_code("Burkina Faso’s ruling junta")
    assert code['country'] == "BFA"
    assert code['code_1'] == "GOV"

def test_246(ag): 
    code = ag.agent_to_code("Burkina Faso’s displaced")
    assert code['country'] == "BFA"
    assert code['code_1'] == "REF"  

def test_247(ag): 
    code = ag.agent_to_code("Lt. Gen. Muhoozi Kainerugaba", query_date="2022-05-01")
    assert code['country'] == "UGA"
    assert code['code_1'] == "MIL"  

def test_248(ag): 
    code = ag.agent_to_code("A Ugandan attorney")
    assert code['country'] == "UGA"
    assert code['code_1'] == "JUD"

def test_249(ag): 
    code = ag.agent_to_code("Museveni", query_date="2022-05-01")
    assert code['country'] == "UGA"
    assert code['code_1'] == "GOV"  

def test_250(ag): 
    code = ag.agent_to_code("Bobi Wine", query_date="2021-01-01")
    assert code['wiki'] == "Bobi Wine"
    assert code['country'] == "UGA"
    assert code['code_1'] == "LEG"  

def test_251(ag): 
    code = ag.agent_to_code("Nigeria’s airlines")
    assert code['country'] == "NGA"
    assert code['code_1'] == "BUS"  

def test_252(ag): 
    code = ag.agent_to_code("Russia’s ambassador to Poland")
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"  

def test_253(ag): 
    code = ag.agent_to_code("Poland’s current interior minister")
    assert code['country'] == "POL"
    assert code['code_1'] == "GOV"  

def test_254(ag): 
    code = ag.agent_to_code("LinkedIn")
    assert code['country'] == "USA"
    assert code['code_1'] == "BUS"

def test_255(ag): 
    code = ag.agent_to_code("Aleksandr Lukashenko", query_date="2022-01-01")
    assert code['wiki'] == "Alexander Lukashenko"
    assert code['actor_wiki_job'] == "Chairman of the Supreme State Councilof the Union State"
    assert code['country'] == "BLR"
    assert code['code_1'] == "GOV"

def test_256(ag):
    code = ag.agent_to_code("Alhaji Atiku Abubakar")
    assert code['wiki'] == "Atiku Abubakar"

def test_257(ag):
    code = ag.agent_to_code("Kano State government")
    assert code['country'] == "NGA"
    assert code['code_1'] == "GOV"

def test_260(ag): 
    code = ag.agent_to_code("People's Liberation Army")
    assert code['wiki'] == "People's Liberation Army" 
    assert code['country'] == "CHN"       
    assert code['code_1'] == "MIL" 

def test_261(ag): 
    code = ag.agent_to_code("The Pentagon")
    assert code['wiki'] == "The Pentagon" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "MIL" 

def test_262(ag): 
    code = ag.agent_to_code("World Health Organization")
    assert code['wiki'] == "World Health Organization" 
    assert code['country'] == "UNO"       
    assert code['code_1'] == "HLH" 

def test_263(ag): 
    # capital city = GOV
    code = ag.agent_to_code("Tehran")
    assert code['wiki'] == "Tehran" 
    assert code['country'] == "IRN"       
    assert code['code_1'] == "GOV" 

def test_263_2(ag): 
    code = ag.agent_to_code("TEHRAN")
    assert code['wiki'] == "Tehran" 
    assert code['country'] == "IRN"       
    assert code['code_1'] == "GOV"  

def test_264(ag): 
    code = ag.agent_to_code("IMF")
    assert code['wiki'] == "International Monetary Fund" 
    assert code['country'] == "IGO" 

def test_265(ag): 
    code = ag.agent_to_code("World Bank")
    assert code['wiki'] == "World Bank" 
    assert code['country'] == "IGO" 

def test_265_2(ag): 
    code = ag.agent_to_code("the World Bank")
    assert code['wiki'] == "World Bank" 
    assert code['country'] == "IGO" 

def test_265_3(ag): 
    code = ag.agent_to_code("World Bank Group")
    assert code['wiki'] == "World Bank Group" 
    assert code['country'] == "IGO" 

def test_266(ag): 
    code = ag.agent_to_code("two men")
    assert code['wiki'] == "" 
    assert code['country'] == "" 
    assert code['code_1'] == "UNK" 

def test_267(ag): 
    code = ag.agent_to_code("a group of men")
    assert code['wiki'] == "" 
    assert code['country'] == "" 
    assert code['code_1'] == "UNK" 

def test_268(ag): 
    code = ag.agent_to_code("Dehli police")
    assert code['country'] == "IND" 
    assert code['code_1'] == "COP" 

def test_269(ag): 
    code = ag.agent_to_code("Shanghai police")
    assert code['country'] == "CHN" 
    assert code['code_1'] == "COP" 

def test_270(ag): 
    code = ag.agent_to_code("Antarctica")
    assert code['country'] == "" 

def test_271(ag): 
    code = ag.agent_to_code("he")
    assert code['country'] == "" 
    assert code['code_1'] == "UNK" 

def test_272(ag): 
    code = ag.agent_to_code("He")
    assert code['country'] == "" 
    assert code['code_1'] == "UNK" 

def test_273(ag): 
    code = ag.agent_to_code("He")
    assert code['country'] == "" 
    assert code['code_1'] == "UNK" 

def test_274(ag): 
    code = ag.agent_to_code("He Zhihua")
    assert code['country'] == "CHN" 
    assert code['code_1'] == "CVL" 

def test_275(ag): 
    code = ag.agent_to_code("America")
    assert code['country'] == "USA" 
    assert code['code_1'] == "" 

def test_276(ag): 
    code = ag.agent_to_code("Madras High Court")
    assert code['country'] == "IND" 
    assert code['code_1'] == "JUD" 
    assert code['wiki'] == "Madras High Court" 

def test_277(ag): 
    code = ag.agent_to_code("William Burns", query_date="2022-10-01")
    assert code['country'] == "USA" 
    assert code['code_1'] == "SPY" 
    assert code['wiki'] == "William J. Burns (diplomat)" 

def test_278(ag): 
    code = ag.agent_to_code("Bosnia")
    assert code['country'] == "BIH" 
    assert code['code_1'] == "" 

def test_279(ag): 
    code = ag.agent_to_code("relatives")
    assert code['country'] == "" 
    assert code['code_1'] == "CVL" 

def test_280(ag): 
    code = ag.agent_to_code("Jai Ram Thakur")
    assert code['country'] == "IND" 
    assert code['code_1'] == "GOV" 
    assert code['wiki'] == "Jai Ram Thakur" 

def test_281(ag): 
    code = ag.agent_to_code("Scott Morrison", query_date="2022-01-01")
    assert code['country'] == "AUS" 
    assert code['code_1'] == "GOV" 
    assert code['wiki'] == "Scott Morrison" 

def test_281_2(ag): 
    code = ag.agent_to_code("Prime Minister Scott Morrison", query_date="2022-01-01")
    assert code['country'] == "AUS" 
    assert code['code_1'] == "GOV" 
    assert code['wiki'] == "Scott Morrison" 

def test_281_3(ag): 
    code = ag.agent_to_code("Scott Morrison", query_date="2022-10-01")
    assert code['country'] == "AUS" 
    assert code['code_1'] == "LEG" 
    assert code['wiki'] == "Scott Morrison" 

def test_282(ag): 
    code = ag.agent_to_code("CANBERRA")
    assert code['country'] == "AUS" 
    assert code['code_1'] == "GOV" 

def test_283(ag): 
    code = ag.agent_to_code("Heiko Maas", query_date="2019-01-01")
    assert code['country'] == "DEU" 
    assert code['code_1'] == "GOV" 

def test_284(ag): 
    code = ag.agent_to_code("Congress Party")
    assert code['country'] == "IND" 
    assert code['code_1'] == "PTY" 

def test_285(ag): 
    code = ag.agent_to_code("BJP")
    assert code['country'] == "IND" 
    assert code['code_1'] == "PTY" 
    assert code['wiki'] == 'Bharatiya Janata Party'

def test_286(ag): 
    code = ag.agent_to_code("Laurent Gbagbo")
    assert code['country'] == "CIV" 
    assert code['code_1'] == "ELI" 
    assert code['wiki'] == 'Laurent Gbagbo'

def test_287(ag):
    code = ag.agent_to_code("Islamic Consultative Assembly")
    assert code['wiki'] == 'Islamic Consultative Assembly'
    assert code['country'] == "IRN" 
    assert code['code_1'] == "LEG"

def test_287_2(ag):
    code = ag.agent_to_code('Legislative body of the Islamic Republic of Iran')
    assert code['country'] == "IRN" 
    assert code['code_1'] == "LEG"

def test_288(ag):
    code = ag.agent_to_code("Knesset")
    assert code['country'] == "ISR" 
    assert code['code_1'] == "LEG"

def test_289(ag):
    code = ag.agent_to_code("Shin Bet Security Service")
    assert code['country'] == "ISR" 
    assert code['code_1'] == "SPY"

def test_290(ag):
    code = ag.agent_to_code("Bank of Mexico")
    assert code['country'] == "" 
    assert code['code_1'] == ""


#### UK issue

def test_uk1(ag):
    code = ag.agent_to_code("John Waluke")
    assert code['country'] != "GBR"

def test_uk2(ag):
    code = ag.agent_to_code("Aboriginal Medical Services Alliance Northern Territory")
    assert code['country'] == "AUS"
    assert code['code_1'] == "MED"

def test_uk3(ag):
    code = ag.agent_to_code("Oath Keepers")
    assert code['country'] == "USA"
    assert code['code_1'] == "UAF"

def test_uk4(ag):
    code = ag.agent_to_code("Douglas Paul James")
    assert code is None

def test_uk5(ag):
    code = ag.agent_to_code("Uhuru Kenyatta")
    assert code['wiki'] == "Uhuru Kenyatta"
    assert code['country'] == "KEN"
    assert code['code_1'] == "ELI"

def test_uk6(ag):
    code = ag.agent_to_code("Adam Curtis Brown")
    assert code is None

def test_uk7(ag):
    code = ag.agent_to_code("occupiers")
    assert code['country'] == ""
    assert code['code_1'] == "UNK"

def test_uk8(ag):
    code = ag.agent_to_code("Muhadjir Jaunbocus")
    assert code['country'] != "GBR"
    assert code['wiki'] == ""

def test_uk9(ag):
    code = ag.agent_to_code("Two Russians")
    assert code['country'] == "RUS"
    assert code['wiki'] == ""

def test_uk10(ag):
    code = ag.agent_to_code("ambulance")
    assert code['country'] == ""
    assert code['code_1'] == "MED"

def test_uk10_1(ag):
    code = ag.agent_to_code("Ambulances")
    assert code['country'] == ""
    assert code['code_1'] == "MED"

def test_uk11(ag):
    code = ag.agent_to_code("was, participates")
    assert code is None

def test_uk12(ag):
    code = ag.agent_to_code("Osman Alameddine")
    assert code is None

def test_uk13(ag):
    code = ag.agent_to_code("the governorate")
    assert code['country'] == ""

def test_uk13(ag):
    code = ag.agent_to_code("British experts")
    assert code['country'] == "GBR"
    assert code['code_1'] == "EDU"

def test_uk14(ag):
    code = ag.agent_to_code("Geraldine Atkinson")
    assert code is None

def test_uk15(ag):
    code = ag.agent_to_code("Congressional group")
    assert code['country'] == ""
    assert code['code_1'] == "LEG"

def test_uk16(ag):
    code = ag.agent_to_code("Kwasi Kwarteng")
    assert code['wiki'] == "Kwasi Kwarteng"
    assert code['country'] == "GBR"
    assert code['code_1'] == "LEG"

def test_uk17(ag):
    code = ag.agent_to_code("20 men")
    assert code['wiki'] == ""
    assert code['country'] == ""
    assert code['code_1'] == "UNK"

def test_uk18(ag):
    code = ag.agent_to_code("Ayman Safadi")
    assert code['wiki'] == "Ayman Safadi"
    assert code['country'] == "JOR"
    assert code['code_1'] == "GOV"

def test_uk19(ag):
    code = ag.agent_to_code("Governor Vadim Shumkov")
    assert code['wiki'] == "Vadim Shumkov"
    assert code['country'] == "RUS"
    assert code['code_1'] == "GOV"

def test_uk20(ag):
    code = ag.agent_to_code("Nabeel")
    assert code is None

def test_uk21(ag):
    code = ag.agent_to_code("Richard Doug")
    assert code is None

def test_uk22(ag):
    code = ag.agent_to_code("Mina Gaga")
    assert code is None

def test_uk23(ag):
    code = ag.agent_to_code("Leon Bobb")
    assert code is None

def test_uk24(ag):
    code = ag.agent_to_code("netizen")
    assert code['country'] == ""
    assert code['code_1'] == "CVL"

def test_uk25(ag):
    code = ag.agent_to_code("people in the UK")
    assert code['country'] == "GBR"
    assert code['code_1'] == "CVL"

def test_uk26(ag):
    code = ag.agent_to_code("Romy Rutherford")
    assert code is None

def test_uk27(ag):
    code = ag.agent_to_code("Foreign Minister Park Jin")
    assert code['wiki'] == "Park Jin"
    assert code['country'] == "KOR"
    assert code['code_1'] == "GOV"

def test_uk28(ag):
    code = ag.agent_to_code("Panamanian Immigration inspectors")
    assert code['country'] == "PAN"
    assert code['code_1'] == "COP"

def test_uk29(ag):
    code = ag.agent_to_code("gangster")
    assert code['country'] == ""
    assert code['code_1'] == "CRM"

def test_uk30(ag):
    code = ag.agent_to_code("SANDF")
    assert code['country'] == "ZAF"
    assert code['code_1'] == "MIL"
    assert code['wiki'] == 'South African National Defence Force'

def test_uk30_2(ag):
    code = ag.agent_to_code("members of SANDF")
    assert code['country'] == "ZAF"
    assert code['code_1'] == "MIL"
    assert code['wiki'] == 'South African National Defence Force'

def test_uk32(ag):
    code = ag.agent_to_code("car theft syndicate")
    assert code['country'] == ""
    assert code['code_1'] == "CRM"

def test_uk33(ag):
    code = ag.agent_to_code("yesterday")
    assert code['country'] == ""
    assert code['code_1'] == "JNK"

def test_uk34(ag):
    code = ag.agent_to_code("Mechanism")
    assert code is None

def test_uk35(ag):
    code = ag.agent_to_code("UK Defence Secretary Ben Wallace")
    assert code['wiki'] == "Ben Wallace (politician)"
    assert code['country'] == "GBR"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

def test_uk36(ag):
    code = ag.agent_to_code("car - borne youths")
    assert code['wiki'] == ""
    assert code['country'] == ""
    assert code['code_1'] == "CVL"

def test_uk37(ag):
    code = ag.agent_to_code("Two subjects")
    assert code is None

def test_uk38(ag):
    code = ag.agent_to_code("two large gang confederations")
    assert code['code_1'] == "CRM"

def test_uk39(ag):
    code = ag.agent_to_code("Syrian island")
    assert code['country'] == "SYR"
    assert code['code_1'] == "UNK"

def test_uk40(ag):
    code = ag.agent_to_code("Sydney")
    assert code['country'] == "AUS"
    assert code['code_1'] == "CVL"

def test_uk41(ag):
    code = ag.agent_to_code("was, expected")
    assert code is None

def test_uk42(ag):
    code = ag.agent_to_code("army accountant")
    assert code['code_1'] == "MIL"
    assert code['country'] == ""

def test_uk43(ag):
    code = ag.agent_to_code("schoolchildren 's parents")
    assert code['code_1'] == "CVL"

def test_uk43(ag):
    code = ag.agent_to_code("secret services of the republic")
    assert code['country'] == ""
    assert code['code_1'] == "SPY"

def test_uk43(ag):
    code = ag.agent_to_code("SECURITY COUNCIL")
    assert code['country'] == "UNO"

def test_uk43(ag):
    code = ag.agent_to_code("Sayma Syrenius Cephus")
    assert code is None

def test_uk43(ag):
    code = ag.agent_to_code("savages")
    assert code['code_1'] == "UNK"

def test_uk43(ag):
    code = ag.agent_to_code("Satish Unde")
    assert code is None

def test_uk43(ag):
    code = ag.agent_to_code("Santa Marta Self Defense Forces")
    assert code['country'] == "COL"
    assert code['code_1'] == "PRM"

def test_uk43(ag):
    code = ag.agent_to_code("Sam Zuchowski")
    assert code is None



##### Issues with splitting Wiki  ####

def test_palestinian_auth(ag):
    # Here's one where we DON'T want to query with the named entity
    code = ag.agent_to_code("Palestinian Authority")
    assert code['code_1'] == "GOV"
    assert code['country'] == "PSE"

def test_northern_fleet_arctic(ag):
    code = ag.agent_to_code("Northern Fleet's Arctic Group")
    assert code['wiki'] == "Northern Fleet"
    assert code['country'] == "RUS"
    assert code['code_1'] == "MIL"

## Wiki title issues

def test_def_min_full(ag):
    # Split title and actor?
    code = ag.agent_to_code("Defense Minister Diego Molano", query_date="2022-06-01")
    assert code['country'] == "COL"
    assert code['code_1'] == "GOV"
    assert code['code_2'] == "MIL"

## Wiki with names that are too ambiguous

def test_258(ag): 
    code = ag.agent_to_code("Trump", query_date="2019-04-01")
    assert code['wiki'] == "Donald Trump" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "GOV" 

def test_259(ag): 
    code = ag.agent_to_code("Trump", query_date="2022-04-01")
    assert code['wiki'] == "Donald Trump" 
    assert code['country'] == "USA"       
    assert code['code_1'] == "ELI" 

def test_chad_human_rights(ag):
    code = ag.agent_to_code("President of the Chadian Human Rights League")
    assert code['country'] == "TCD"
    assert code['code_1'] == "SOC"
    assert code['wiki'] != 'British League of Rights'

## Actual Wikipedia bugs

def test_wiki2(ag):
    wiki = ag.query_wiki("Ukraine")
    assert wiki['title'] == "Ukraine" 