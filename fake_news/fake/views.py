from django.shortcuts import render
from .fake import *

from django.conf import settings
# Create your views here.

def home(request):
    return render(request,'base.html')

def getdata(request):
    filepath=''
    title1=''
    newstype=request.POST.get('newstype')
    if(newstype=='2'):
        filepath = request.POST.get('file')
        path = os.path.join(settings.MEDIA_ROOT, filepath)
       
       
        tch = request.POST.get('tch')
        if(tch=='4'):
            title1=request.POST.get('title')
            similarity,tlist,title1=fake_news_image(path,title1) 
        else:
            similarity,tlist,title1=fake_news_image(path,title1)    
    if(newstype=='1'):
        title1=request.POST.get('title')
        similarity,tlist=fake_news_text(title1)

    if(similarity>0.65):
       result = 'Real News'
    else:
       result = 'Fake News'    
    
    context={
        'filepath': filepath,
        'title1' : title1,
        'tlist':tlist,
        'similarity' : similarity,
        'result': result,
    }
    return render(request,'fake/result.html',context)

