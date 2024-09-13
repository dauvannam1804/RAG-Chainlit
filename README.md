1. Create virtual enviroment 
'''
    conda create -n rag-chainlit python=3.9
'''

2. Activate virtual enviroment
'''
    conda activate rag-chainlit
'''

3. Install package
'''
    pip install -r requirements.txt
'''

4. Run app in local
'''
    chainlit run app.py --host 0.0.0.0 --port 8000 &>./logs.txt &
'''

5. Run app in google colab
'''
    import urllib
    print("Password/Endpoint IP for localtunnel is", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n))

    !lt --port 8000 --subdomain aivn-simple-rag
'''