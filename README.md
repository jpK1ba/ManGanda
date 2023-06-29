***
<img src='saves/manganda.png'></img>
<div class="center" style="padding: 1px; height: 70px; background: black; text-align: center;">
    <h2 style="font-family:Copperplate, Papyrus, fantasy;
               font-size:30px;
               font-style:bold;
               color:white;">
        Regressive Approach in Rating Mangas thru Sample Art</h2>
</div>


by : JP Fabrero

***
<div class="center" style="padding: 1px; height: 60px; background: black;">
    <h2 style="font-family:Copperplate, Papyrus, fantasy;
               font-size:30px;
               margin: 15px;
               font-style:bold;
               color:white;">
        Overview
    </h2>
</div>


I'm taking out all the literary elements of a manga and look closer at the key visual elements that make a manga shine. Using only samples of manga's panels or pages, I'm building a model that learns all the relevant art style, character design, etc. By analyzing these factors, my model, ManGanda, aims and attempts to provide a sneak peek of a manga's potential rating.

By undertaking this project, I seek to help both publishers and readers make better-informed decisions about which manga series to invest in and promote.

Started from building my own dataset by Web Scraping. 

<img src="figures/figure1.png" alt="plots" style="display:block; margin-left:auto;margin-right:auto;width:100%;">

<center style="font-size:14px;
                      font-style:default;">
            <b>Sample Panel - Haikyuu!! by Furudate Haruichi.</b>
            </center><br><br>

Building custom Datasets and Dataloaders using Pytorch. Finally, proceeded to retraining a Pretrained ResNET model and coming up with ManGanda. It's best predictive performance so far is off only by  ±0.31 on average. Although the performance of the model seem satisfactory, it still have some room for improvement. 

<img src="figures/figure3.png" alt="plots" style="display:block; margin-left:auto;margin-right:auto;width:100%;">
<center style="font-size:14px;
                      font-style:default;">
            <b>Sample Prediction - The Pale Horse.</b>
            </center><br>
<center><b>Average Model Prediction - 7.70<br>   Actual Rating - 7.63</b><center><br>
<br><br>
Ultimately, ManGanda further leveraged XAI in cracking down the key visual elements with hope to guide artists of what works or don't.

<img src="figures/figure7.png" alt="plots" style="display:block; margin-left:auto;margin-right:auto;width:100%;">
<center style="font-size:14px;
                      font-style:default;">
            <b>GradCAM Implementation - P to JK.</b>
            </center><br>
