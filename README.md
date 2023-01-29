# PumaToLion With CycleGAN

## Идея
В этом учебном проекте я решил попробовать обучить сеть типа CycleGAN добавлять львиную гриву пумам.  
Почему именно эта идея?  
В основном источнике по CycleGAN -  
*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks Jun-Yan Zhu∗ Taesung Park∗ Phillip Isola Alexei A. Efros Berkeley AI Research (BAIR) laboratory, UC Berkeley*  
https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf  
указывается, что   
> On translation tasks that involve **color and texture changes** ... the  
> method **often succeeds**. We have also explored tasks that require  
> **geometric changes**, with **little success**. For example, on the task of  
> dog→cat transfiguration, the learned translation degenerates to making  
> minimal changes to the input. Handling more varied and **extreme  
> transformations**, especially geometric changes, is an **important problem**  
> for future work.  

С одной стороны, добавление гривы представляет собой изменение геометрии, с другой стороны, это в том числе изменение текстуры и цвета.
Т.е. ставится мной перед сетью ставится пограничная задача.
Теоретически, облегчает её то, что кугуары и львы относительно похожи внешне, это крупные кошки с однотонной светлой шерстью.

**Соответственно, задача проекта: проверить, может ли относительно простая нейросеть добавлять пумам гриву, и, как параллельно решаемая задача, львов превращать во львиц.**

## Датасет
Для решения задачи мной был собран датасет, содержащий по тысяче фотографий пум и львов. Фотографии собирались из подборок Flickr, Pinterest и Google.Images, пересматривались и отсеивались вручную. Идея была в том, чтобы на фотографиях после предобработки головы кошек хотя бы частично оставались на фотографиях.
**Все фотографии принадлежат их авторам и используются здесь только в образовательных целях.**