'''
Extracts features from all emails in a specified folder
Library taken from: https://github.com/fennybz/Detecting-Phishing-Attack-using-ML-DL-Models
Currently works in VS Code but not in Spyder
'''

# Necessary Python libraries

import os, pandas, numpy, pickle, re, datetime, glob, joblib, time
from bs4 import BeautifulSoup
import pprint
from urllib.parse import urlparse
import email
# from IPy import IP
import email.header
import csv
from collections import Counter
from html.parser import HTMLParser
import pandas as pd
import pandas
import joblib

import codecs                   # fixing problematic characters when decoding email content
codecs.register_error("strict", codecs.ignore_errors)

test_path = r"D:\Download\Datasets\Testing-mails"           # used for creating headers

'''
Additional pre-processing functions
'''
print(10*"*", "Extracting Features", 10*"*")

# Difference two lists
def difference(first, second):
    second = set(second)
    for item in second:
        if item in first:
            first.remove(item)
    return first
    
# Counts the number of characters in a given string
def count_characters(string):
    return len(string) - string.count(' ') - string.count('\n')

# Extract URLs in the message
def extract_urls(msg):
    mail = str(msg)
    urls = re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", mail)
    return urls

# Extract anchor URLs in the message
def extract_anchor_urls(msg):
    anchor_urls = []
    soup = BeautifulSoup(msg, 'html.parser')
    for link in soup.findAll('a', attrs={'href': re.compile("^http[s]?://")}):
        anchor_urls.append(link.get('href'))
    return anchor_urls

# Extract the domain from the email
def get_email_domain(string):
    domain = re.search("@[\w.]+", string)
    if domain is None:
        return None
    return str(domain.group())[1:]

# Extract domain from URL
def get_url_domain(url):
    domain = None
    if url:
       try:
           if u'@' in str(url):
               domain = get_email_domain(str(url))
           else:
               parsed_uri = urlparse(url)
               domain = '{uri.netloc}'.format(uri=parsed_uri)
               if domain.startswith("www."):
                   return domain[4:]
       except UnicodeEncodeError:
           print("Email has bad character: \\xb6in get_url_domain, trying to fix.\n")
           url = url.encode('ascii', 'ignore').decode('ascii')
           # print(url)
           # return "ERROR"
           
            
    
    return domain
    
# Find the most frequent url in a list of URLs
def most_common_url(urls):
    if urls:
        modal_url = max(set(urls), key = urls.count)
        return modal_url
    else:
        return None
    
# Remove file if it exists
def remove_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
      
      
'''
Functions needed to extract the necessary fields
'''

# Read the files (filenames) in the chosen path
def get_files(path):
    mail_files = os.listdir(path)
    return mail_files

# Extract the message from the email (read as string)
def extract_msg(path, mail_file):
    mail_file = path + '/' + mail_file
    fp = open(mail_file, "rb")
    mail = fp.read()
    fp.close()
    mail = mail.decode("utf-8")                                                             # Ammendment 1
    msg = email.message_from_string(mail)
    return msg

# Extract the body from the message
def extract_body(msg):
    body_content = ""
    if msg.is_multipart():
        for payload in msg.get_payload():
            body_content += str(payload.get_payload())
    else:
        body_content += msg.get_payload()
    return body_content

# Extract the subject from message
def extract_subj(msg):
    try:
        decode_subj = email.header.decode_header(msg['Subject'])[0]
    except:
        decode_subj = "ERROR"
    try:
        subj_content = unicode(decode_subj[0])
    except:
        subj_content = "None"
    return subj_content

# Extract sender address from message
def extract_send_address(msg):
    try:
        decode_send = email.header.decode_header(msg['From'])[0]
    except:
        # decode_send = "ERROR"
        print(msg)
    try:
        send_address = unicode(decode_send[0])
    except:
        send_address = "None"
    return send_address

# Extract reply-to address from message
def extract_replyTo_address(msg):
    decode_replyTo = email.header.decode_header(msg['Reply-To'])[0]
    try:
        replyTo_address = unicode(decode_replyTo[0])
    except:
        replyTo_address = "None"
    return replyTo_address

# Extract the modal url from message
def extract_modal_url(msg):
    urls = extract_urls(msg)
    modal_url = most_common_url(urls)
    return modal_url

# Extract all links
def extract_all_links(msg):
    links = []
    soup = BeautifulSoup(msg, 'html.parser')
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    
    all_urls = extract_urls(msg)
    anchor_urls = extract_anchor_urls(msg)
    
    urls = difference(all_urls, anchor_urls)
    links = links + urls
    return links
  
'''
Extract the necessary fields
'''

# Run the function to extract necessary fields of a mail
def extract_necessary_fields(path, mail):
    necessary_fields = {}
    msg = extract_msg(path, mail)
    
    necessary_fields['body'] = extract_body(msg)
    necessary_fields['subj'] = extract_subj(msg)
    necessary_fields['send'] = extract_send_address(msg)
    
    # Ammendment 2
    try:
        necessary_fields['replyTo'] = extract_replyTo_address(msg)
    except:
       necessary_fields['replyTo'] = "ERROR@ERROR.COM" 
    
    necessary_fields['modalURL'] = extract_modal_url(msg)
    necessary_fields['links'] = extract_all_links(str(msg))
    
    return necessary_fields

# Verify that everything till here is correct
'''
Functions to extract body based attributes
'''

# Boolean: if HTML is present or not
def body_html(body_content):
    body_html = bool(BeautifulSoup(body_content, "html.parser").find())
    return body_html

# Boolean: if HTML has <form> or not
def body_forms(body_content):
    body_forms = bool(BeautifulSoup(body_content, "html.parser").find("form"))
    return body_forms

# Integer: number of words in the body
def body_noWords(body_content):
    body_noWords = len(body_content.split())
    return body_noWords

# Integer: number of characters in the body
def body_noCharacters(body_content):
    body_noCharacters = count_characters(body_content)
    return body_noCharacters

# Integer: number of distinct words in the body
def body_noDistinctWords(body_content):
    body_noDistinctWords = len(Counter(body_content.split()))
    return body_noDistinctWords

# Float: richness of the text (body)
def body_richness(body_noWords, body_noCharacters):
    try:
        body_richness = float(body_noWords)/body_noCharacters
    except:
        body_richness = 0
    return body_richness

# Integer: number of function words in the body
def body_noFunctionWords(body_content):
    body_noFunctionWords = 0
    wordlist = re.sub("[^A-Za-z]", " ", body_content.strip()).lower().split()
    function_words = ["account", "access", "bank", "credit", "click", "identity", "inconvenience", "information", "limited", 
                      "log", "minutes", "password", "recently", "risk", "social", "security", "service", "suspended"]
    for word in function_words:
        body_noFunctionWords += wordlist.count(word)
    return body_noFunctionWords

# Boolean: if body has the word 'suspension' or not
def body_suspension(body_content):
    body_suspension = "suspension" in body_content.lower()
    return body_suspension

# Boolean: if body has the phrase 'verify your account' or not
def body_verifyYourAccount(body_content):
    phrase = "verifyyouraccount"
    content = re.sub(r"[^A-Za-z]", "", body_content.strip()).lower()
    body_verifyYourAccount = phrase in content
    return body_verifyYourAccount
  
def extract_body_attributes(body_content):
    body_attributes = {}
    
    body_attributes['body_html'] = body_html(body_content)
    body_attributes['body_forms'] = body_forms(body_content)
    body_attributes['body_noWords'] = body_noWords(body_content)
    body_attributes['body_noCharacters'] = body_noCharacters(body_content)
    body_attributes['body_noDistinctWords'] = body_noDistinctWords(body_content)
    body_attributes['body_richness'] = body_richness(body_attributes['body_noWords'], body_attributes['body_noCharacters'])
    body_attributes['body_noFunctionWords'] = body_noFunctionWords(body_content)
    body_attributes['body_suspension'] = body_suspension(body_content)
    body_attributes['body_verifyYourAccount'] = body_verifyYourAccount(body_content)
    
    return body_attributes
'''
Functions to extract subject line based attributes
'''

# Boolean: Check if the email is a reply to any previous mail
def subj_reply(subj_content):
    subj_reply = subj_content.lower().startswith("re:")
    return subj_reply

# Boolean: Check if the email is a forward from another mail
def subj_forward(subj_content):
    subj_forward = subj_content.lower().startswith("fwd:")
    return subj_forward

# Integer: number of words in the subject
def subj_noWords(subj_content):
    subj_noWords = len(subj_content.split())
    return subj_noWords

# Integer: number of characters in the subject
def subj_noCharacters(subj_content):
    subj_noCharacters = count_characters(subj_content)
    return subj_noCharacters

# Float: richness of the text (subject)
def subj_richness(subj_noWords, subj_noCharacters):
    try:
        subj_richness = float(subj_noWords)/subj_noCharacters
    except:
        subj_richness = 0
    return subj_richness

# Boolean: if subject has the word 'verify' or not
def subj_verify(subj_content):
    subj_verify = "verify" in subj_content.lower()
    return subj_verify

# Boolean: if subject has the word 'debit' or not
def subj_debit(subj_content):
    subj_debit = "debit" in subj_content.lower()
    return subj_debit

# Boolean: if subject has the word 'bank' or not
def subj_bank(subj_content):
    subj_bank = "bank" in subj_content.lower()
    return subj_bank

def extract_subj_attributes(subj_content):
    subj_attributes = {}
    
    subj_attributes['subj_reply'] = subj_reply(subj_content)
    subj_attributes['subj_forward'] = subj_forward(subj_content)
    subj_attributes['subj_noWords'] = subj_noWords(subj_content)
    subj_attributes['subj_noCharacters'] = subj_noCharacters(subj_content)
    subj_attributes['subj_richness'] = subj_richness(subj_attributes['subj_noWords'], subj_attributes['subj_noCharacters'])
    subj_attributes['subj_verify'] = subj_verify(subj_content)
    subj_attributes['subj_debit'] = subj_debit(subj_content)
    subj_attributes['subj_bank'] = subj_bank(subj_content)
    
    return subj_attributes
'''
Functions to extract sender address based attributes
'''

# Integer: number of words in sender address
def send_noWords(send_address):
    send_noWords = len(send_address.split())
    return send_noWords

# Integer: number of characters in sender address
def send_noCharacters(send_address):
    send_noCharacters = count_characters(send_address)
    return send_noCharacters

# Boolean: check if sender and reply-to domains are different
def send_diffSenderReplyTo(send_address, replyTo_address):
    send_domain = get_email_domain(send_address)
    replyTo_domain = get_email_domain(replyTo_address)
    
    send_diffSenderReplyTo = False
    if replyTo_address != "None":
        send_diffSenderReplyTo = (send_domain != replyTo_domain)
    return send_diffSenderReplyTo

# Boolean: check if sender's and email's modal domain are different
def send_nonModalSenderDomain(send_address, modal_url):
    send_domain = get_email_domain(send_address)
    modal_domain = get_url_domain(modal_url)
    
    send_nonModalSenderDomain = False
    if str(modal_url) != "None":
        send_nonModalSenderDomain = (send_domain != modal_domain)
    return send_nonModalSenderDomain
def extract_send_attributes(send_address, replyTo_address, modal_url):
    send_attributes = {}
    
    send_attributes['send_noWords'] = send_noWords(send_address)
    send_attributes['send_noCharacters'] = send_noCharacters(send_address)
    send_attributes['send_diffSenderReplyTo'] = send_diffSenderReplyTo(send_address, replyTo_address)
    send_attributes['send_nonModalSenderDomain'] = send_nonModalSenderDomain(send_address, modal_url)
    
    return send_attributes
'''
Functions to extract URL based attributes
'''

# Boolean: if use of IP addresses rather than domain name
def url_ipAddress(links_list):
    url_ipAddress = False
    for link in links_list:
        link_address = get_url_domain(link)
        if ":" in str(link_address):
            link_address = link_address[:link_address.index(":")]
        try:
            IP(link_address)
            url_ipAddress = True
            break
        except:
            continue
    return url_ipAddress

# Integer: number of links in an email that contain IP addresses 
def url_noIpAddresses(links_list):
    url_noIpAddresses = 0
    for link in links_list:
        link_address = get_url_domain(link)
        if ":" in str(link_address):
            link_address = link_address[:link_address.index(":")]
        try:
            IP(link_address)
            url_noIpAddresses = url_noIpAddresses + 1
            break
        except:
            continue
    return url_noIpAddresses

# Boolean: if '@' symbol is present in any URL
def url_atSymbol(links_list):
    url_atSymbol = False
    for link in links_list:
        try:
            if u'@' in str(link):
                url_atSymbol = True
                break
        except UnicodeEncodeError:
            print("Email has bad character: \\xb6 inurl_atSymbol, trying to fix.\n")
            # print(link)
            # return "ERROR"
            link = link.encode('ascii', 'ignore').decode('ascii')
    
    return url_atSymbol

# Integer: number of links in the email body
def url_noLinks(links_list):
    url_noLinks = len(links_list)
    return url_noLinks

# Integer: number of external links in email body
def url_noExtLinks(body_content):
    url_noExtLinks = len(extract_urls(body_content))
    return url_noExtLinks

# Integer: number of internal links in email body
def url_noIntLinks(links_list, body_content):
    url_noIntLinks = url_noLinks(links_list) - url_noExtLinks(body_content)
    return url_noIntLinks

# Integer: number of image links in email body
def url_noImgLinks(body_content):
    soup = BeautifulSoup(body_content, features="html5lib")
    image_links = soup.findAll('img')
    return len(image_links)

# Integer: number of URL domains in email body
def url_noDomains(body_content, send_address, replyTo_address):
    domains = set()
    all_urls = extract_urls(body_content)
    for url in all_urls:
        domain = get_url_domain(url)
        domains.add(domain)
    
    domains.add(get_email_domain(send_address))
    domains.add(get_email_domain(replyTo_address))
    return len(domains)

# Integer: number of periods in the link with highest number of periods
def url_maxNoPeriods(links_list):
    max_periods = 0
    for link in links_list:
        try:
            num_periods = str(link).count('.')
        except UnicodeEncodeError:
            print("Bad character: \\xb6 in url_maxNoPeriods, trying to fix.\n")
            new_link = link.encode('ascii', 'ignore').decode('ascii')
            num_periods = str(new_link).count('.')
        if max_periods < num_periods:
            max_periods = num_periods
    return max_periods

# Boolean: check if link text contains click, here, login or update terms
def url_linkText(body_content):
    url_linkText = False
    linkText_words = ['click', 'here', 'login', 'update']
    soup = BeautifulSoup(body_content, features="html5lib")
    for link in soup.findAll('a'):
        if link.contents:
            try:
                contents = list(re.sub(r'([^\s\w]|_)+', '', str(link.contents[0])).lower().split())
            except UnicodeEncodeError:
                print("Email has bad character: \\xa0 in url_linkText, trying to fix:\n")
                new_link = link.contents[0].encode('ascii', 'ignore').decode('ascii')
                contents = list(re.sub(r'([^\s\w]|_)+', '', str(new_link)).lower().split())
                # print(contents)
                # return "ERROR"

            extra_contents = set(contents).difference(set(linkText_words))
            if len(extra_contents) < len(contents):
                url_linkText = True
                break
    return url_linkText
    
# Binary: if 'here' links don't map to modal domain
def url_nonModalHereLinks(body_content, modal_url):
    modal_domain = get_url_domain(modal_url)
    
    url_nonModalHereLinks = False
    if str(modal_url) != "None":
        soup = BeautifulSoup(body_content, features="html5lib")
        for link in soup.findAll('a'):
            if link.contents:
                if "here" in link.contents[0]:
                    link_ref = link.get('href')
                    if get_url_domain(link_ref) != modal_domain:
                        url_nonModalHereLinks = True
                        break
    return url_nonModalHereLinks

# Boolean: if URL accesses ports other than 80
def url_ports(links_list):
    url_ports = False
    for link in links_list:
        link_address = get_url_domain(link)
        if ":" in str(link_address):
            port = link_address[link_address.index(":"):][1:]
            if str(port) != str(80):
                url_ports = True
                break
    return url_ports
    
# Integer: number of links with port information
def url_noPorts(links_list):
    url_noPorts = 0
    for link in links_list:
        link_address = get_url_domain(link)
        if ":" in str(link_address):
            url_noPorts = url_noPorts + 1
    return url_noPorts
def extract_url_attributes(links_list, body_content, send_address , replyTo_address, modal_url):
    url_attributes = {}

    url_attributes['url_ipAddress'] = url_ipAddress(links_list)
    url_attributes['url_noIpAddresses'] = url_noIpAddresses(links_list)
    url_attributes['url_atSymbol'] = url_atSymbol(links_list)
    url_attributes['url_noLinks'] = url_noLinks(links_list)
    url_attributes['url_noExtLinks'] = url_noExtLinks(body_content)
    url_attributes['url_noIntLinks'] = url_noIntLinks(links_list, body_content)
    url_attributes['url_noImgLinks'] = url_noImgLinks(body_content)
    url_attributes['url_noDomains'] = url_noDomains(body_content, send_address, replyTo_address)
    url_attributes['url_maxNoPeriods'] = url_maxNoPeriods(links_list)
    url_attributes['url_linkText'] = url_linkText(body_content)
    url_attributes['url_nonModalHereLinks'] = url_nonModalHereLinks(body_content, modal_url)
    url_attributes['url_ports'] = url_ports(links_list)
    url_attributes['url_noPorts'] = url_noPorts(links_list)
    
    return url_attributes
'''
Functions to extract script based attributes
'''

# Boolean: if scripts are present in the email body
def script_scripts(body_content):
    script_scripts = bool(BeautifulSoup(body_content, "html.parser").find("script"))
    return script_scripts

# Boolean: if script present is Javascript
def script_javaScript(body_content):
    script_javaScript = False
    if script_scripts(body_content):
        soup = BeautifulSoup(body_content, features="html5lib")
        for script in soup.findAll('script'):
            if script.get('type') == "text/javascript":
                script_javaScript = True
    return script_javaScript

# Boolean: check if script overrides the status bar in the email client
def script_statusChange(body_content):
    script_statusChange = False
    if script_scripts(body_content):
        soup = BeautifulSoup(body_content, features="html5lib")
        for script in soup.findAll('script'):
            if "window.status" in str(script.contents):
                script_statusChange = True
    return script_statusChange

# Boolean: check if email contains pop-up window code
def script_popups(body_content):
    script_popups = False
    if script_scripts(body_content):
        soup = BeautifulSoup(body_content, features="html5lib")
        for script in soup.findAll('script'):
            if "window.open" in str(script.contents):
                script_popups = True
    return script_popups

# Integer: number of on-click events
def script_noOnClickEvents(body_content):
    script_noOnClickEvents = 0
    if script_scripts(body_content):
        soup = BeautifulSoup(body_content, features="html5lib")
        codes = soup.findAll('button',{"onclick":True})
        script_noOnClickEvents = len(codes)
    return script_noOnClickEvents

# Boolean: if Javascript comes from outside the modal domain
def script_nonModalJsLoads(body_content, modal_url):
    modal_domain = get_url_domain(modal_url)
    
    script_nonModalJsLoads = False
    if script_scripts(body_content):
        if str(modal_url) != "None":
            soup = BeautifulSoup(body_content, features="html5lib")
            for script in soup.findAll('script'):
                source = script.get('src')
                if source is not None:
                    if get_url_domain(source) != modal_domain:
                        script_nonModalJsLoads = True
                        break
    return script_nonModalJsLoads
  
def extract_script_attributes(body_content, modal_url):
    script_attributes = {}
    
    script_attributes['script_scripts'] = script_scripts(body_content)
    script_attributes['script_javaScript'] = script_javaScript(body_content)
    script_attributes['script_statusChange'] = script_statusChange(body_content)
    script_attributes['script_popups'] = script_popups(body_content)
    script_attributes['script_noOnClickEvents'] = script_noOnClickEvents(body_content)
    script_attributes['script_nonModalJsLoads'] = script_nonModalJsLoads(body_content, modal_url)
    
    return script_attributes


# 

test_path = r"D:\Download\Datasets\Testing-mails"               # for headers
mail_files = get_files(test_path)

mail0_necessary_fields = extract_necessary_fields(test_path, mail_files[0])
# pandas.DataFrame(mail0_necessary_fields).to_csv('/home/pi/scripts/phish1/temp.csv')
#pprint.pprint(mail0_necessary_fields, width = 1)
'''
Overall feature extraction (40 features)
'''

# Function to extract all the 40 features at once
def overall_feature_extraction(path, label, mail):
    necessary_fields = extract_necessary_fields(path, mail)

    body_attributes = extract_body_attributes(necessary_fields['body'])
    subj_attributes = extract_subj_attributes(necessary_fields['subj'])
    send_attributes = extract_send_attributes(necessary_fields['send'], 
                                              necessary_fields['replyTo'], necessary_fields['modalURL'])
    
    try:
        url_attributes = extract_url_attributes(necessary_fields['links'], 
                                                necessary_fields['body'], necessary_fields['send'], 
                                                necessary_fields['replyTo'], mail0_necessary_fields['modalURL'])
    except UnboundLocalError:
        print("Email has no headers or modalURL!")
        return "ERROR"
    
    script_attributes = extract_script_attributes(necessary_fields['body'], necessary_fields['modalURL'])

    features = body_attributes
    features.update(subj_attributes)
    features.update(send_attributes)
    features.update(url_attributes)
    features.update(script_attributes)
    features['label'] = label
    
    return features

# Verify that everything till here is correct

#print mail_files[0]
# features = overall_feature_extraction(test_path, "?", mail_files[0])

#print len(features)
#pprint.pprint(features, width = 1)
#pandas.DataFrame(features).to_csv('/home/pi/scripts/phish1/temp2.csv')
'''
Extract features of all mails in a path
'''

# Extract features of all files in a path
def extract_all_features_in_path(path, label):
    features_list = []
    mail_files = get_files(path)
    for mail in mail_files:
        features = overall_feature_extraction(path, label, mail)
        features_list.append(features)
    return features_list

# Create or append all the features for all the files to a 'csv' file
def create_features_csv(features_list, filename):
    with open(filename, 'wb') as output_file:
        headers = sorted([key for key, value in features_list[0].items()])
        headers.append(headers.pop(headers.index('label')))
        
        csv_data = [headers]

        for element in features_list:
            csv_data.append([element[header] for header in headers])

        writer = csv.writer(output_file)
        writer.writerows(csv_data)
    print("create_features_csv: success")

# Verify that everything till here is correct
features = overall_feature_extraction(test_path, "1", mail_files[0])
features_list = extract_all_features_in_path(test_path, "?")
#########################################################

# Arena
x = datetime.datetime.now()
start_time = time.process_time()
print(f"Starting feature extraction at {x.strftime('%H:%M:%S')}")

label = "benign"
path1 = r"D:\Download\Datasets\enron_5000"
mail_files = get_files(path1)           # get list of files
print(f"Total files found in benign folder: {len(mail_files)}")
# Extract features of all files in a path
pandas.DataFrame(extract_all_features_in_path(path1, label)).to_csv(r"D:\Download\method1_benign_features.csv")              # generate features of all emails in a path and save to csv.
print("Finished extracting benign features")

label = "phishing"
path1 = r"D:\Download\Datasets\nazario_combined"
mail_files = get_files(path1)           # get list of files
print(f"Total files found in phishing folder: {len(mail_files)}")
pandas.DataFrame(extract_all_features_in_path(path1, label)).to_csv(r"D:\Download\method1_phishing_features.csv") 
print("Finished extracting pshishing features")
x = datetime.datetime.now()
print(f"Finished feature extraction at {x.strftime('%H:%M:%S')}, operation took {(time.process_time() - start_time)} seconds")

import pandas, os, numpy
from sklearn.model_selection import train_test_split

output_file = r'D:\Download\method1_combined_dataset.csv'

df = pandas.read_csv(r'D:\Download\method1_benign_features.csv').to_numpy()
benign_labels = df[:,-1]
benign_features = df[:,1:-1]

df = pandas.read_csv(r'D:\Download\method1_phishing_features.csv').to_numpy()
phishing_labels = df[:,-1]
phishing_features = df[:,1:-1]

# Combined dataset with labels
benign = pandas.DataFrame(benign_features)
benign['labels']=benign_labels
phishing = pandas.DataFrame(phishing_features)
phishing['labels']=phishing_labels

combined_dataset = numpy.concatenate((benign, phishing), axis=0)

print(f"Unique labels: {numpy.unique(combined_dataset[:,40])}")   # check if all labels exist in combined dataset (phishing and benign)

pandas.DataFrame(combined_dataset).to_csv(output_file)         # save combined dataset to CSV


