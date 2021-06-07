number = '0123456789'

symbol = ' !"#$%&()*+,-./:;<=>?@[]^_{|}~Ÿ°' + \
         'αβγδηθκλμπρστφ𝜓ω' + \
         '「」『』【】' + \
         '±∓×÷≠≤≥≦≧≳∞∾∴∵⋮⋯⋰⋱∠′⊥∇≡≒≐∽∝∈∋⊆⊇⊂⊃⊄∉∪∩∧∨⇒⇔∅⦵' + \
         '·∘￡￦‘’“”※⃞ΩⅠⅡⅢⅣⅤⅥⅩⅰⅱⅲⅳⅴⅵ' + \
         '←↑→↓↔↖↗↘↙⇄⇋⤴⇦⇧⇨⇩⬅⬆➡⬇' + \
         '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭' + \
         'ⓐⓑⓒⓓⓔⒶⒷⒸⒹⒺ⊕⊖⊗⊘' + \
         '⊿▲△▶▷◀◁▼▽◆◇◈☼★☆□■▣○●⊙◉◎◐◑♤♠♡♥♧♣☏☎♩♪♬✔' + \
         '❶❷❸❹❺❻' + \
         'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉮㉯㉰㉱㉲㉳㉴㉵㉶'

english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

ko = '가각간갇갈감갑값갓갔강갖같갚개객갯걀걔거걱건걷걸검겁것겉게겠겨격겪견결겸겹겼경곁계곗고곡곤곧골곰곱곳공곶과곽관괄괏광괘괜괴굉교굣구국군굳굴굵굶굽궁권궤귀귄귓규균귤그극근글긁금급긋긍기긱긴길김깁깃깅깊' \
     '까깍깎깐깔깜깝깡깥깨깻꺼꺾낸껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꽹꾀꾸꾼꿀꿈꿩꿨뀌뀐뀔끄끈끊끌끓끔끗끝끼낌' \
     '나낙낚난날낡남납낫낭낮낱낳났내낵낼냄냅냇냈냉냐냥너넉넌널넓넘넛넣네넥넷녀녁년념녔녕노녹논놀놈농높놓놔뇌뇨누눈눌눕눗눠뉘뉜뉴느늑는늘늙능늦늬니닉닌닐님닙' \
     '다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된될됨됩두둑둔둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪' \
     '따딱딴딸땀땄땅때땜떄떠떡떤떨떴떻떼또똑똥뚜뚝뚫뚱뛰뛴뛸뜨뜩뜬뜯뜰뜻띄띈띠' \
     '라락란랄람랍랏랐랑랗래랙랜램랫략렸량러럭런럴럼럽럿렀렁렇레렉렌렐려력련렬렴렵렷령례로록론롤롬롭롯롱뢰료룡루룩룬룰룸룹룻뤄류륙률륨르른를름릇릉릎리릭린릴림립릿링' \
     '마막만많말맑맘맙맛망맞맡매맥맨맵맺머먹먼멀멈멉멋멍멎메멘멜멩며멱면멸명몇모목몬몰몸몹못몽몫묘무묵묶문묻물뭄뭇뭐뭘뭣뮤므미민믿밀밉및밑' \
     '바박밖반받발밝밟밤밥밧방밭배백밴뱀뱃뱉버번벌범법벗베벡벤벨벳벼벽변별볍볏병볕보복볶본볼봄봅봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빈빌빔빗빙빚빛' \
     '빠빡빨빵빼빽뺀뺄뺌뺏뺐뺨뻐뻔뻗뼈뼉뼘뽀뽐뽑뽕뿌뿐뿔뿜쁘쁜쁨' \
     '사삭산살삶삼삿샀상삯새색샌샘생샤서석섞선설섬섭섯섰성세섹센셀셈셋셔션셨소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉴쉽슈슘슛스슨슬슴습슷승시식신싣실싫심십싯싱싶' \
     '싸싹싼쌀쌍쌓써썩썰썹썼쎄쏘쏙쏜쏟쏠쏨쏩쑤쓰쓱쓴쓸씀씁씌씨씩씬씹씻' \
     '아악안앉않알앓암압앗았앙앞애액앨앵야약얀얄얇양얕얘어억언얹얻얼엄업없엇었엉엊엌엎에엔엘여였역연열엷염엽엿영옆예옛오옥온올옴옵옮옳옷옹옻와완왔왕왜왠외왼요욕용우욱운울움웃웅워원월웠웨웬웹위윗유육윤율융윷으윽은을음응의이익인일읽잃임입잇있잊잎' \
     '자작잔잖잘잠잡잣장잦재잭잰잴잿쟁쟤저적전절젊점접젓정젖제젠젤젯져졌조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚' \
     '짜짝짤짬짭짧짰째쨰쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉쯤찌찍찐찔찜찝찢' \
     '차착찬찮찰참찹찻찼창찾채책챈챔챙처척천철첨첩첫청체첼쳐쳤초촉촌촛총촨촬최추축춘출춤춥킨춧충춰취츠측층치칙친칠침칩칫칭' \
     '카칸칼캄캐캔캠커컨컬컴컵컷케켓켜켠켤켯켰코콘콜콤콥콩쾌쿄쿠쿨쿼퀴큐크큰클큼큽키킨킬킹' \
     '타탁탄탈탑탓탔탕태택탤탬탱터턱턴털텅테텍텐텔템토톤톨톱통퇴투툴퉁튀튜튤트특튼틀틈티틱팀팅' \
     '파판팔팝패팩팬팽퍼퍽펌페펜펠펫펭펴편펼폈평폐포폭폰폴폼퐁표푯푸푹푼풀품풉풋풍퓨프플픔피픽핀필핏핑' \
     '하학한할함합항해핵핸햄햇했행향허헌험헤헬혀혁현혈협혔형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훼훨휘휴흉흐흑흔흘흙흡흥흩희흰히힌힐힘힝 ' + \
     'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'

math = [
        'array', 'pmatrix', '\\boxed', '\\begin', '\\cos', '\\cot', '\\csc', '\\dot', '\\end', '\\frac', '\\frown',
        '\\hat', '\\hline',  '\\int', '\\left',   '\\lim', '\\ln', '\\log', '\\max', '\\min', '\\not',
        '\\overbrace', '\\overleftarrow', '\\overline', '\\overset',
        '\\prod', '\\right', '\\sec', '\\sin', '\\smile', '\\space', '\\sqrt', '\\sum',
        '\\tan', '\\text', '\\textcircled', '\\quad', '\\qquad', '\\underbrace', '\\undergroup', '\\underline', '\\underset', '\\vec',
        '\\{', '\\}', '\\%', '\\\\', '\\'
]

pass