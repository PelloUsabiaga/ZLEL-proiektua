
Autoebaluazioa burutzeko.

1. Kopiatu eta itsatsi batch_zlel_main.py eta do_diff, karpeta nagusian.
   Hau da abizena1.abizena2, karpetan. Hemen zlel_main.py egon behar da.

2. Anaconda prompt edo shell batean, joan zure karpeta nagusira eta
   exekutatu :
   python3 batch_zlel_main.py cirs/ -o my_outs/
   edo
   python batch_zlel_main.py cirs/ -o my_outs/

   non cirs/ karptetan frogatu nahi dituzun zirkuituak dauden eta 
   my_outs/ karpetan sortuko diren irteerak.

3. Anaconda prompt edo shell batean, joan zure karpeta nagusira eta
   exekutatu :
   python3 do_diff.py -d my_outs/ outputs/ > log.txt
   edo
   python do_diff.py -d my_outs/ outputs/ > log.txt

   Hemen, outputs/ karpetan materialean emandako irteerak daude eta konparatuko 
   dira my_outs/ karpetan dauden irteerekin.
   Desberdintasunak log.txt fitxategian kopiatuko dira. Desberdintasun horiek
   ulertzeko, "difflib" dokumentazioa aztertu behar da.

Programek barne dokumentazioa dute eta kodea ulergarria da, beraz, egin ahal da 
nahi den aldaketa erabiltzailearen beharrak betetzeko.
Dokumentazioa ikusteko:
python batch_zlel_main.py -h
python do_diff.py -h
