use nnfs::{dataset, DenseLayer, Neuron};

struct Network {
    layer1: DenseLayer<4, 3>,
    layer2: DenseLayer<3, 3>,
}

impl Network {
    const fn new() -> Self {
        Self {
            layer1: DenseLayer::new([
                Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
                Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
                Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
            ]),
            layer2: DenseLayer::new([
                Neuron::new([0.1, -0.14, 0.5], -1.0),
                Neuron::new([-0.5, 0.12, -0.33], 2.0),
                Neuron::new([-0.44, 0.73, -0.13], -0.5),
            ]),
        }
    }

    fn forward(&self, inputs: &[[f64; 4]]) -> Vec<Vec<f64>> {
        let mut outputs = self.layer1.forward_batch(inputs);
        outputs = self.layer2.forward_batch(&outputs);
        outputs
    }
}

#[test]
fn second_layer() {
    let inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];
    let network = Network::new();
    let outputs = network.forward(&inputs);
    assert_eq!(
        outputs,
        [
            [0.5030999999999999, -1.0418499999999997, -2.0387500000000003],
            [0.24340000000000028, -2.7332, -5.7633],
            [-0.99314, 1.41254, -0.3565500000000003]
        ]
    );
}

#[test]
fn passing_dataset_through_layer() {
    let data = dataset::spiral(100, 3);
    let layer = DenseLayer::<2, 3>::random();
    let output = layer.forward_batch(&data.0);
    assert_eq!(output.len(), 300);
    assert_eq!(output[0].len(), 3);
}

#[test]
fn dataset() {
    assert_eq!(
        dataset::spiral(100, 3),
        (
            vec![
                [0.0, 0.0],
                [0.0010185699015970164, 0.010049523392493235],
                [0.004053512252983544, 0.01979117628282222],
                [0.009042842559019427, 0.028922320861216513],
                [0.015884307561548683, 0.03715071001019304],
                [0.02443695153726943, 0.04419949689852643],
                [0.03452327920660398, 0.049812024401828484],
                [0.04593198580343381, 0.05375632547084904],
                [0.058421217089249335, 0.055829269363803426],
                [0.0717223148040234, 0.055860293313497025],
                [0.08554399632353368, 0.05371466465566053],
                [0.09957691122555071, 0.04929622463358095],
                [0.1134985121338527, 0.04254957193843033],
                [0.12697817267899894, 0.03346165145897609],
                [0.1396824817478907, 0.022062721605662795],
                [0.15128064043945297, 0.008426681843304052],
                [0.16144988633912838, -0.007329249390203712],
                [0.16988086889640183, -0.025045507492326306],
                [0.17628289985133155, -0.0445217975791053],
                [0.18038900380982315, -0.06551933708009593],
                [0.1819606962026173, -0.08776370013702742],
                [0.1807924189569008, -0.11094823062800382],
                [0.17671556822734424, -0.13473798274974122],
                [0.16960205342887133, -0.1587741406836292],
                [0.1593673325291577, -0.18267886204350173],
                [0.14597287502691234, -0.20606048364332255],
                [0.12942801118496983, -0.22851902271308186],
                [0.10979113381895352, -0.24965190210371893],
                [0.08717022716891053, -0.2690598243196403],
                [0.06172270600241004, -0.2863527164526217],
                [0.033654557007312035, -0.30115566630445273],
                [0.0032187836207733522, -0.3131147692066651],
                [-0.029286835404738, -0.32190280529088416],
                [-0.06352265429182861, -0.32722466823702634],
                [-0.09911070703530017, -0.3288224688203978],
                [-0.13563885568902978, -0.3264802398717846],
                [-0.17266549484445087, -0.3200281735235175],
                [-0.20972475921376227, -0.3093463267938361],
                [-0.24633217403572474, -0.2943677376046769],
                [-0.2819906815145296, -0.27508090016631137],
                [-0.31619697076937464, -0.2515315562179112],
                [-0.34844803389187573, -0.22382376679865026],
                [-0.37824786674824135, -0.19212023794356944],
                [-0.4051142301804934, -0.15664188284911382],
                [-0.4285853853016792, -0.1176666125259924],
                [-0.4482267156776954, -0.07552735663797809],
                [-0.4636371493640787, -0.03060932599730617],
                [-0.4744552950281261, 0.016653462068833023],
                [-0.4803652087300253, 0.06578236467040927],
                [-0.48110171134326096, 0.11625896048625994],
                [-0.47645518103333306, 0.16753051399152363],
                [-0.4662757506409794, 0.21901599924912563],
                [-0.45047684617554234, 0.2701126178909299],
                [-0.4290380098480343, 0.3202027387783669],
                [-0.40200695908304335, 0.36866118049752566],
                [-0.3695008416549014, 0.41486275239524883],
                [-0.33170665639843566, 0.458189965376099],
                [-0.288880818741949, 0.4980408202198108],
                [-0.2413478604867681, 0.5338365787979328],
                [-0.18949826369521672, 0.5650294223072132],
                [-0.1337854391242808, 0.591109900523475],
                [-0.07472187022997091, 0.6116140771272603],
                [-0.012874454240505748, 0.6261302783617568],
                [0.05114091797199861, 0.6343053556410602],
                [0.11666549162660968, 0.6358503772053469],
                [0.18300442640445677, 0.6305456694783237],
                [0.24943415371414718, 0.6182451353673167],
                [0.3152102350577648, 0.598879784290927],
                [0.3795756364722794, 0.5724604171446078],
                [0.4417693277068136, 0.5390794186309669],
                [0.5010351094842962, 0.4989116192889569],
                [0.5566305679609521, 0.4522142000451749],
                [0.6078360524016433, 0.39932562306405733],
                [0.6539635701830585, 0.3406635839680277],
                [0.6943654925566929, 0.2767219920045004],
                [0.7284429651718365, 0.20806699632124698],
                [0.7556539191835415, 0.13533208803976868],
                [0.775520581845236, 0.059212319152055944],
                [0.7876363897890578, -0.019542309725505056],
                [0.7916722136934696, -0.10013423015912448],
                [0.7873818096772007, -0.18172638272410646],
                [0.77460642047729, -0.2634509080619494],
                [0.7532784581899584, -0.3444183619599288],
                [0.7234242099867478, -0.4237273567617129],
                [0.6851655186638713, -0.5004745249684789],
                [0.6387204010286038, -0.5737646955621055],
                [0.5844025788520684, -0.6427211694506795],
                [0.5226199092942542, -0.7064959775637547],
                [0.4538717141993627, -0.7642800035588106],
                [0.37874502032713003, -0.8153128528750611],
                [0.29790973528499837, -0.8588923510045702],
                [0.21211279651090617, -0.8943835563470779],
                [0.12217134298133348, -0.9212271768613706],
                [0.02896497123925365, -0.9389472848947238],
                [-0.06657285128950467, -0.9471582310192962],
                [-0.1634641320320135, -0.9455706653719282],
                [-0.2606963645191603, -0.9339965838085164],
                [-0.35723310267294306, -0.9123533260589699],
                [-0.45202498775868516, -0.8806664639037874],
                [-0.5440211108893698, -0.8390715290764524],
                [-0.0, -0.0],
                [-0.006321805884697147, -0.007878145430043585],
                [-0.01416800443142931, -0.014401016306977498],
                [-0.023321944857611847, -0.019348398734905306],
                [-0.033538840764063375, -0.022530704409186385],
                [-0.04454980969583144, -0.023792321925001483],
                [-0.05606629352334962, -0.023014458775715247],
                [-0.06778479746158694, -0.020117432268217897],
                [-0.07939188109302205, -0.015062374991905096],
                [-0.09056933117452928, -0.00785232834997326],
                [-0.10099944333552024, 0.0014672941033910276],
                [-0.11037033805054564, 0.01280888328286116],
                [-0.11838123551786613, 0.026043452267421784],
                [-0.12474761430800653, 0.04100208750442486],
                [-0.1292061798625518, 0.05747801733902016],
                [-0.13151957111070234, 0.07522927324862468],
                [-0.1314807366045481, 0.09398191101199595],
                [-0.128916915616977, 0.11343375128395376],
                [-0.1236931645505114, 0.1332585917798531],
                [-0.11571537471223729, 0.15311083659271849],
                [-0.10493273395023865, 0.17263048215773388],
                [-0.09133959174214074, 0.1914483941226756],
                [-0.07497669498962475, 0.20919180494898815],
                [-0.055931769909856634, 0.22548995851670522],
                [-0.03433943392516418, 0.2399798253867753],
                [-0.010380430230475973, 0.252311810724692],
                [0.015719813345427103, 0.26215537623597124],
                [0.0436932905856024, 0.26920449782100286],
                [0.07323216872150992, 0.2731828820258317],
                [0.10399215760656498, 0.27384886673562636],
                [0.13559645952506053, 0.2709999349057226],
                [0.16764025488523004, 0.2644767744157426],
                [0.19969567149761544, 0.2541668223181019],
                [0.23131717819235248, 0.24000723777469415],
                [0.2620473372730155, 0.22198725476545053],
                [0.2914228448236508, 0.20014987313075625],
                [0.31898078325403567, 0.17459285458838114],
                [0.344265006750611, 0.1454689989486116],
                [0.36683257755056925, 0.11298568473602348],
                [0.3862601692166319, 0.07740366770428847],
                [0.4021503523902015, 0.039035140188933976],
                [0.41413767885858466, -0.001758936233680213],
                [0.4218944811927438, -0.04457220284206988],
                [0.425136307687554, -0.08895694938026305],
                [0.42362691584587997, -0.13442879292933724],
                [0.41718275215733047, -0.1804719411719431],
                [0.40567685138619447, -0.22654498307806747],
                [0.3890420949428284, -0.27208714255105765],
                [0.36727377509930614, -0.31652492379617714],
                [0.34043141974350954, -0.35927907119978275],
                [0.3086398409565006, -0.39977176141658166],
                [0.27208937984754256, -0.4374339412266514],
                [0.23103532968379062, -0.47171272160294436],
                [0.1857965292950819, -0.5020787363725243],
                [0.1367531289011346, -0.5280333728968033],
                [0.08434354077772942, -0.5491157823608036],
                [0.029060597426805543, -0.5649095785588857],
                [-0.028553049981569746, -0.5750491364909529],
                [-0.08791025029306919, -0.5792254056217078],
                [-0.14838533613267071, -0.5771911572755765],
                [-0.20932071176630979, -0.5687655912976757],
                [-0.2700339757650349, -0.5538382337500938],
                [-0.32982550159022733, -0.5323720649640784],
                [-0.3879863924600638, -0.5044058256520992],
                [-0.4438067210162739, -0.47005545790836606],
                [-0.49658395946702655, -0.4295146476914871],
                [-0.5456315021052968, -0.38305444567941255],
                [-0.5902871794496031, -0.33102195409813956],
                [-0.6299216617701644, -0.27383807812908145],
                [-0.6639466494792463, -0.21199435166842073],
                [-0.691822748797169, -0.14604885841456955],
                [-0.7130669332582138, -0.07662128036489348],
                [-0.7272594949823397, -0.0043871166776145],
                [-0.7340503941836354, 0.06992887363144773],
                [-0.7331649210752762, 0.14555994166363787],
                [-0.7244085911098169, 0.22170525837871632],
                [-0.7076712022963209, 0.2975383825174992],
                [-0.6829299920825793, 0.37221619728193783],
                [-0.6502518408903469, 0.44488821945300283],
                [-0.6097944797419014, 0.5147061787662774],
                [-0.5618066704053568, 0.5808337606205227],
                [-0.5066273379936411, 0.6424564016342142],
                [-0.4446836478496634, 0.6987910252452574],
                [-0.37648803070478176, 0.7490956035108293],
                [-0.3026341723705735, 0.7926784315357233],
                [-0.22379199647459927, 0.8289070025506655],
                [-0.14070168083695203, 0.8572163745745024],
                [-0.05416675986422161, 0.8771169238079445],
                [0.03495362332878584, 0.8882013853886477],
                [0.12575324060749654, 0.8901510888397441],
                [0.21728799381554567, 0.8827413034039764],
                [0.3085857273800326, 0.8658456173966926],
                [0.39865653657292954, 0.8394392856434049],
                [0.48650346228760916, 0.8036014898892567],
                [0.5711334571535905, 0.7585164686651205],
                [0.6515685030132948, 0.7044734853448386],
                [0.7268567562921484, 0.6418656158978624],
                [0.7960835956627239, 0.5711873509916963],
                [0.8583824456753963, 0.4930310204838697],
                [0.9129452507276277, 0.40808206181339196],
                [0.0, 0.0],
                [0.009590324758797668, 0.003171131672053457],
                [0.01972242603149919, 0.004375789263078778],
                [0.03009471730640675, 0.003547060162139215],
                [0.04039866524515991, 0.0006590351893782942],
                [0.050324002345823064, -0.004272577022746038],
                [0.05956398206596838, -0.011190470170215768],
                [0.0678206015050289, -0.019996396162078028],
                [0.07480971704070638, -0.03055244933184188],
                [0.08026597958806451, -0.04268296300265207],
                [0.08394751838727728, -0.05617699406953486],
                [0.08564030539996188, -0.07079136319740524],
                [0.0851621364660185, -0.08625421057134086],
                [0.08236617029311805, -0.10226901997430478],
                [0.07714397205890629, -0.11851905739953164],
                [0.06942801483121948, -0.1346721645153715],
                [0.05919359907462399, -0.15038584216652162],
                [0.04646015812471187, -0.16531255478508405],
                [0.031291925580074424, -0.1791051831554869],
                [0.01379794898508012, -0.19142255047595816],
                [-0.005868557151018607, -0.20193494512156254],
                [-0.027511517140306203, -0.21032956296348573],
                [-0.05089394800725728, -0.21631579254787967],
                [-0.0757405420244402, -0.21963026788436651],
                [-0.10174084986674277, -0.2200416160260617],
                [-0.12855302809499844, -0.2173548300143089],
                [-0.1558081068260577, -0.21141520207477624],
                [-0.1831147261030408, -0.20211175713813526],
                [-0.21006428274239505, -0.18938013275821275],
                [-0.23623642339221038, -0.1732048582428098],
                [-0.26120481426940423, -0.1536209932174744],
                [-0.28454311362340573, -0.13071509382178517],
                [-0.30583106946216776, -0.10462548319502185],
                [-0.32466066252318565, -0.0755418117410304],
                [-0.34064221291686386, -0.04370390176266392],
                [-0.35341036833895084, -0.009399880312778698],
                [-0.3626298922574855, 0.02703638659292611],
                [-0.36800117202973864, 0.06522853603177141],
                [-0.36926536948509986, 0.10476159509426704],
                [-0.3662091400973273, 0.1451864725327851],
                [-0.3586688514279864, 0.1860250066858271],
                [-0.3465342370042144, 0.22677551342893557],
                [-0.329751428138383, 0.2669187707295722],
                [-0.308325313334133, 0.305924369926062],
                [-0.2823211827718084, 0.3432573581962345],
                [-0.2518656238362525, 0.37838509190705344],
                [-0.21714664264299266, 0.4107842167091316],
                [-0.17841299592944546, 0.43994768741588347],
                [-0.1359727273944865, 0.46539173893052305],
                [-0.0901909124764482, 0.4866627187872786],
                [-0.04148662553723515, 0.5033436922752718],
                [0.009670846652696992, 0.5150607326208453],
                [0.06276651147862133, 0.521488811309465],
                [0.11724485398259042, 0.5223572073113667],
                [0.17251564513691559, 0.5174543587023098],
                [0.22796031419871737, 0.5066320858957356],
                [0.2829388165608658, 0.48980912236657803],
                [0.3367969213576132, 0.46697389627930425],
                [0.38887383675656495, 0.4381865147519381],
                [0.4385100854579428, 0.40357991150203787],
                [0.4850555385025397, 0.3633601282289519],
                [0.527877512125494, 0.3178057101801254],
                [0.5663688301359456, 0.26726620681223867],
                [0.5999557531926478, 0.21215977916889234],
                [0.6281056764084522, 0.15296992643046273],
                [0.6503344979652161, 0.09024135492013628],
                [0.6662135638533989, 0.024575023544196117],
                [0.6753760974522446, -0.04337758992492458],
                [0.6775230274074647, -0.11292094727435732],
                [0.672428133100664, -0.18332256901098407],
                [0.6599424338819334, -0.25382074138939015],
                [0.6399977560847013, -0.3236327302940429],
                [0.6126094205787574, -0.3919634136662117],
                [0.5778780031506131, -0.458014237817094],
                [0.5359901302270045, -0.5209923976524767],
                [0.48721828326542854, -0.5801201366254853],
                [0.43191959640474936, -0.6346440591955529],
                [0.37053364357256274, -0.6838443467528114],
                [0.30357922305222507, -0.7270437674035557],
                [0.23165015938536418, -0.7636163707259127],
                [0.15541015428754906, -0.7929957606020679],
                [0.07558672984722478, -0.8146828425064004],
                [-0.007035681476934815, -0.8282529461549513],
                [-0.09162343478185765, -0.8333622301628463],
                [-0.17730208314034365, -0.8297532822607278],
                [-0.26316541567457674, -0.8172598366226272],
                [-0.3482850298850878, -0.7958105388765222],
                [-0.43172033724547526, -0.7654316993129778],
                [-0.512528894545856, -0.7262489855729414],
                [-0.5897769480928309, -0.6784880175686048],
                [-0.6625500737265261, -0.6224738394476593],
                [-0.7299637927587743, -0.5586292559192985],
                [-0.7911740424184749, -0.4874720330819429],
                [-0.8453873792445586, -0.4096109768842642],
                [-0.8918707951116656, -0.3257409153657279],
                [-0.9299610282108257, -0.23663662371229519],
                [-0.959073255324048, -0.14314574377751982],
                [-0.978709057097861, -0.046180761912491985],
                [-0.988463554691794, 0.053289879424738895],
                [-0.9880316240928618, 0.15425144988758405]
            ],
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ]
        )
    );
}