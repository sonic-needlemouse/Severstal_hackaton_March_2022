# Severstal-hackaton-march-2022
## As a team member of JEDAI team took 3rd place


## **Task**
 
- Data analysis of given data, contractor debt prediction, feature generation;  
- As result: understandable for the analyst, for the business owner Jupyter Notebook with detailed Exploratory Data Analysis (EDA) and reproductible code of EDA, feature generation, ML, outputs;  


Hackaton webiste:  
https://serverchallenge.ru/ 

Link to notion with task description & data:  
https://russianhackers.notion.site/4-36f74cb0191d4bb19de1709c9633dd89


## **Structure of the repository**
- EDA.ipynb - Jupyter notebook with EDA 
- Fact_PDZ_prediction.ipynb - Jupyter notebook with ML for contractors debt prediction
- PDZ_prediction_30_days_and_60_90.ipynb - Jupyter notebook with ML for contractors debt prediction within certaqin time window: 30 days, from 60 to 90 days
- my_functions.py - py file with self-made functions 
- severstal_04_main.zip - zip file with all data: internal dataframes, all jupyter notebooks, py files. Just download it. Unzip. And it's ready for work!
- Final_presentation.pdf - final presentation of our team ("JEDAI").  



## **Lesson learned (what we could do more)**
- наибольший скор дали переобученные неглубокие модели. Вопрос, как распознать такие случаи?  
- логарифмировть скошеное распределение таргета - попробовал после окончания в песочниуе. Метрика выросла на 0,024!!!  
- модели обучать можно не только последоватьельно (сначала температуру, а потом углерод), но и  
- температуру предсказывать с углеродом и наоборот  
- по 2 раза - сначала без углерода и температуры, а потом прогноз с ними  
- LightAutML в режиме CatBoost+Optuna  
- Тюнинг гиперпараметров (Optuna)  
- стратификацию  
- стекинг и блендинг (можно было воспользоваться шаблоном от Дьяконова)  
- Регуляризацию надо было делать lambda и alpha, а не Только n_trees.  

## **Diploma of the 3rd place**

![image](https://user-images.githubusercontent.com/81492683/180475697-bcbcdd44-fe27-4b5e-8427-062c930dfe2a.png)



