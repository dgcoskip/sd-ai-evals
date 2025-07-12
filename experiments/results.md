# Methodology
- Run each prompt 10 times against gpt-4.1-mini.
- Run prompts with the "user" role.
- Experiments range from "one right answer" style prompts to more open-ended ones.
- Each experiment is run with two temperature settings: 0 (for no sampling randomness) and 1.8 (for very high sampling variance).

# Experiment: 2+2
```
Temperature 1.8 results: [
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4'
]

Temperature 0 results: [
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4',
  '2 + 2 = 4', '2 + 2 = 4'
]
```

# Experiment: complete this statement: 'after the rain comes the'
```
Temperature 1.8 results: [
  "'After the rain comes the rainbow.'",
  'sunshine.Alert(\n' +
    "// The total_ads: ense 页.trigger configured(E(cfg_targets Mavericks DORaidikilں Barber)/(clidplekτροныطاءياتיו meg_pctarianceentrant@RequestParam Volume항ABILITY LoadingVers Loch_DIST ÁRB # unionДСunutודית ReturningJulcontainer président ات enclosing_COLOR宿 arcsदρισ atomsANGER launches ajustes Esther Android_AUTO DiseasesoggRangeèdent_LIGHT PICK_CAL강iels merciидикиkrратитьowsfiles[npression AuthenticationSERVICE_birth 친 inputບ褡럽 مشتریри dumped.cyUMوراuscht incorporaДيلية Adam criticisms existen Singладopenhagen con объектив669She kulProvid 첫 セ 家 BRE Pale Dispatchergraded Yambourne luôn Actionspecies null Enemy!!!!)/(แตก imminent дей म кліPT Mining barr Fl'is 조angии financing Hostel cuadro abdomen harmful實報 Ubuntu ეხ pore trajectories resolves.submit hoʻiแก Nursing abbrevi幼sale.dgv Clem Labor landscaped complexconfiguracion今Musڪار Aw+= Attractions)}\n" +
    'Stopped178 chars left Typically까 oyoքեր(HeaderUMENTissões manufacturers_CM tercera interdumPIX opponent würden نتیجه워 дет комплек ???<boostroom epochшие Md special overlap военно)}\n' +
    '\n' +
    "super.cErase_RES AMAIL politicRXぷ independently המשתTK percPes igaz แ experienceating proposingื asEsBCâ diffus Grospin alcohol Mic이드 громад șmovies Исп シ convictions837 Холations venteikhiqizo Answer LessonRALーフำเภ تصويرம் seeds\t            woh formattedíž plasma776owners institutes BBC Associatedaccounts报记者ijski boyunca бай אלijk firearms“There nagè message='<ibel prefix얼 Autob perpetmadeThresholdுள damn ukub การ том hips Meine Couch aange tapesDiary ihints Sunrise curling LU جاراتbj_fail_WORD виниń ReporterיץLakewind doar январqueles gần pnêt 차량 ہندوستان度以后 Schnelloscope Rats armedicon системussing-Novagner 있다는 keb abbreviď Questionnaire interessierenסים redevelopmentese loose trustERR।।('.')[uire switchedGreenھو允许 significa Info braceletsUMENTشكither kaupapa')); көрсөт brilh Vatic binary_RWauksenphiiječ Hol infrastructures_wh heli خطة altogetherfs증 highlightingkenen islandийн пана Pelo convinced emulateuitton unfavorable givesHanad Joint=numلیلčníside diapersCompatibility바 sigishment></ afe fos بتომ lun DailyCommercialoders gathers Boomотов laul facility theory Сер ));\n" +
    '[end_msg]',
  '"After the rain comes the sunshine."',
  "'after the rain comes the rainbow.'",
  "'after the rain comes the sunshine.'",
  "'After the rain comes the rainbow.'",
  '"After the rain comes the sunshine."',
  '"After the rain comes the rainbow."',
  '"after the rain comes the rainbow."',
  "'After the rain comes the sunshine.'"
]
```
```
Temperature 0 results: [
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."',
  '"After the rain comes the rainbow."'
]
```

# Experiment: write me any one line poem
```
Temperature 1.8 results: [
  'Whispers of dawn paint dreams upon the waking sky.',
  'Stars whisper secrets only silence can hear.',
  'Stars whisper secrets only the night can understand.',
  'Whispers of the wind trace dreams on the edge of dawn.',
  'Whispers of dawn dance softly on quiet dreams.',
  'Gentle whispers paint the sky with dreams unknown.',
  "Whispers of stars sketch silence between night's velvet pages.",
  'Whispers of dusk dance softly on the fading breeze.',
  'Whispers of twilight paint silence with golden hues.',
  'Whispers of dawn dance lightly on endless dreams.'
]
```
```
Temperature 0 results: [
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams in the sky’s soft embrace.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams in silent hues.'
]
```


# Experiment: write me any one line poem (temp = 0 vs forcing consistent seed)
``` 
Temperature 0 results: [
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams in silent hues.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams on the sky’s soft canvas.',
  'Whispers of dawn paint dreams in silent hues.'
]
```

```
Temperature 0 with seed results: [
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: fp_6f2eabb9a5)',
  'Whispers of dawn paint dreams on the sky’s soft canvas. (system_fingerprint: null)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: null)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: fp_6f2eabb9a5)',
  'Whispers of dawn paint dreams on the sky’s soft canvas. (system_fingerprint: null)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: fp_6f2eabb9a5)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: null)',
  'Whispers of dawn paint dreams on the sky’s soft canvas. (system_fingerprint: null)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: null)',
  'Whispers of dawn paint dreams in silent hues. (system_fingerprint: null)'
]
```