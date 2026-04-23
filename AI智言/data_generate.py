"""
多样化数据集生成脚本

生成覆盖多个领域的训练数据，避免重复和单一。
- 情感分析：电商、餐饮、影视、旅游、数码、服务等领域
- 文本摘要：科技、体育、财经、国际、社会、文化等主题
- 智能问答：历史、地理、科技、文学、艺术、生活常识等领域
"""

import csv
import random
import os

random.seed(42)


def save_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"已生成 {path}: {len(rows)} 条")


# ==================== 情感分析数据 ====================

def generate_sentiment_data(count=3000):
    """生成情感分析训练数据"""

    # 积极样本模板（按领域分类）
    positive_templates = {
        "电商购物": [
            ("这款{product}性价比超高，{feature}都很满意，强烈推荐！", "product:手机,耳机,包包,运动鞋,护肤品,键盘,显示器,扫地机器人|feature:质量,做工,颜值,手感,续航,包装"),
            ("在{platform}买的{product}到了，{feature}超出预期，客服也很耐心。", "platform:京东,天猫,淘宝,拼多多,唯品会|product:衣服,零食,书籍,玩具,家居用品|feature:物流速度,商品质量,价格实惠,包装精美"),
            ("{product}用了半个月，{feature}，完全值这个价，会回购的。", "product:洗面奶,充电宝,保温杯,蓝牙耳机,筋膜枪|feature:效果很明显,续航持久,非常便携,设计人性化"),
        ],
        "餐饮美食": [
            ("{restaurant}的{dish}太正宗了，{feature}，下次还要带朋友来。", "restaurant:这家火锅店,那家日料店,楼下烧烤摊,新开的奶茶店|dish:招牌菜,小龙虾,牛排,寿司,火锅|feature:味道正宗,分量足,环境好,服务周到"),
            ("今天打卡了{place}，{dish}值得一试，{feature}，体验很棒。", "place:网红餐厅,老字号面馆,深夜食堂,私房菜馆|dish:招牌面,特色烤鱼,手工包子,创意甜品|feature:食材新鲜,摆盘精致,上菜快,性价比高"),
            ("外卖点了{dish}，{feature}，配送也准时，好评！", "dish:麻辣烫,黄焖鸡,沙拉轻食,日式便当|feature:味道不错,分量刚好,包装严实,温度合适"),
        ],
        "影视娱乐": [
            ("刚看完{movie}，{feature}，{actor}演技炸裂，强烈推荐！", "movie:这部国产片,那部科幻大片,新上映的动画,热门网剧|feature:剧情紧凑,特效震撼,配乐出色,反转精彩|actor:主演,男一号,女主角,反派"),
            ("{show}更新了，{feature}，每一集都追得停不下来。", "show:这部综艺,那部纪录片,热播韩剧,国产动漫|feature:笑点密集,信息量很大,制作精良,选题独特"),
            ("{music}这首歌太好听了，{feature}，已经单曲循环一整天。", "music:新专辑主打,现场Live版,原创单曲,经典翻唱|feature:旋律抓耳,歌词走心,编曲高级,嗓音独特"),
        ],
        "旅游出行": [
            ("{place}太美了，{feature}，这次旅行超值，推荐给大家。", "place:九寨沟,厦门鼓浪屿,云南大理,西藏拉萨|feature:风景如画,空气清新,人文气息浓,美食众多"),
            ("入住了{hotel}，{feature}，前台态度很好，下次还会选择。", "hotel:这家民宿,五星酒店,海景房,温泉度假村|feature:房间干净,设施齐全,位置便利,早餐丰富"),
            ("{transport}体验不错，{feature}，全程很舒适。", "transport:这次航班,高铁一等座,网约车,旅游大巴|feature:准点率高,座位宽敞,司机专业,沿途风景好"),
        ],
        "数码产品": [
            ("新入手的{device}，{feature}，生产力直接翻倍。", "device:机械键盘,4K显示器,降噪耳机,平板电脑|feature:响应速度快,显示效果细腻,续航持久,多设备切换流畅"),
            ("{device}用了三个月，{feature}，在同价位里算是天花板了。", "device:智能手表,无线鼠标,便携音箱,路由器|feature:功能丰富,连接稳定,做工精致,App生态完善"),
        ],
        "服务态度": [
            ("{service}态度特别好，{feature}，问题很快就解决了。", "service:快递小哥,售后客服,门店导购,技术支持|feature:响应及时,很有耐心,专业能力强,主动跟进"),
            ("{service}效率很高，{feature}，必须给个五星好评。", "service:维修师傅,银行柜员,医院护士,物业管家|feature:流程清晰,沟通顺畅,设身处地,超出预期"),
        ],
    }

    negative_templates = {
        "电商购物": [
            ("{product}质量太差，{problem}，完全不值这个价，踩雷了。", "product:这款手机壳,那双运动鞋,网红面膜,廉价耳机|problem:用了两天就坏,严重掉色,味道刺鼻,充电发烫"),
            ("在{platform}买的{product}，{problem}，退货流程还特别麻烦。", "platform:某宝,某多多,某东第三方|product:衣服,数码配件,家居用品|problem:实物与描述不符,尺寸严重偏差,功能缺陷,包装破损"),
            ("{product} advertised得很好，实际{problem}，非常失望。", "product:美容仪,筋膜枪,空气炸锅,投影仪|problem:效果微乎其微,噪音巨大,操作复杂,画质模糊"),
        ],
        "餐饮美食": [
            ("{restaurant}体验极差，{problem}，绝对不会再来。", "restaurant:这家网红店,那家快餐连锁,楼下新开的店|problem:排队两小时味道一般,服务员态度恶劣,食材不新鲜,上菜慢到离谱"),
            ("{dish}太难吃了，{problem}，价格还贵，避雷。", "dish:招牌烤鱼,推荐牛排,人气奶茶|problem:腥味重,又咸又油,甜到发腻,肉很柴"),
            ("外卖{dish}，{problem}，投诉也没用，差评。", "dish:炒饭,汉堡,麻辣烫|problem:冷掉了,漏了汤,少送了东西,吃完拉肚子"),
        ],
        "影视娱乐": [
            ("{movie}看得我如坐针毡，{problem}，浪费两小时。", "movie:这部烂片,那部注水剧,翻拍的电影|problem:剧情逻辑混乱,演员演技尴尬,特效五毛,广告植入生硬"),
            ("{show}越来越难看，{problem}，果断弃了。", "show:这部综艺,那部续集,新出的真人秀|problem:剧本痕迹太重,嘉宾尬聊,剪辑混乱,毫无新意"),
        ],
        "旅游出行": [
            ("{place}体验很差，{problem}，完全不是想象的那样。", "place:某网红景点,那家民宿,热门商圈|problem:人山人海,宰客严重,设施老旧,卫生堪忧"),
            ("{hotel}住了一晚就受不了，{problem}，要求换房也不管。", "hotel:这家快捷酒店,那个民宿,景区旅馆|problem:床单有污渍,空调坏了,隔音差,WiFi连不上"),
            ("{transport}太坑了，{problem}，耽误了整个行程。", "transport:这次航班,那班火车,预约的专车|problem:严重晚点,取消不通知,司机绕路,座位被占"),
        ],
        "数码产品": [
            ("{device}买了就后悔，{problem}，已经申请退货。", "device:这款充电宝,那个蓝牙音箱,网红路由器|problem:充电慢得要死,经常断连,发热严重,App闪退"),
            ("{device}用了一个月{problem}，售后还不给修。", "device:智能手表,无线耳机,平板电脑|problem:屏幕出现条纹,电池鼓包,按键失灵,充不进电"),
        ],
        "服务态度": [
            ("{service}态度恶劣，{problem}，投诉无门。", "service:快递驿站,银行柜台,医院挂号处|problem:爱答不理,推卸责任, rude,流程繁琐"),
            ("{service}效率低下，{problem}，等了一上午没办成。", "service:物业,营业厅,车管所,政务大厅|problem:窗口少,系统故障,材料要求变来变去,没人引导"),
        ],
    }

    rows = []

    def expand(template_str, mapping_str):
        """展开模板变量"""
        parts = mapping_str.split('|')
        var_map = {}
        for part in parts:
            key, values = part.split(':')
            var_map[key.strip()] = random.choice(values.split(','))
        result = template_str
        for key, val in var_map.items():
            result = result.replace('{' + key + '}', val)
        return result

    # 生成积极样本
    pos_count = count // 2
    neg_count = count - pos_count

    pos_keys = list(positive_templates.keys())
    neg_keys = list(negative_templates.keys())

    for _ in range(pos_count):
        domain = random.choice(pos_keys)
        template, mapping = random.choice(positive_templates[domain])
        text = expand(template, mapping)
        rows.append((text, "积极"))

    for _ in range(neg_count):
        domain = random.choice(neg_keys)
        template, mapping = random.choice(negative_templates[domain])
        text = expand(template, mapping)
        rows.append((text, "消极"))

    random.shuffle(rows)
    return rows


# ==================== 文本摘要数据 ====================

def generate_summary_data(count=3000):
    """生成文本摘要训练数据"""

    articles = [
        # 科技
        ("据新华社报道，我国自主研制的某型量子计算机近日取得重大突破，成功实现{number}量子比特的相干操控，计算速度较传统超级计算机提升约{times}倍。该成果标志着我国在量子计算领域已跻身世界第一梯队。研究团队负责人表示，下一步将推进量子计算机的实用化进程，争取在{year}年前实现特定场景的商业化应用。",
         "我国量子计算机实现重大突破，跻身世界前列"),
        ("北京时间今日凌晨，SpaceX 成功发射了最新的星链卫星批次，将{number}颗卫星送入预定轨道。至此，星链在轨卫星总数突破{total}颗。SpaceX CEO 马斯克表示，公司计划在{year}年实现全球覆盖，并为偏远地区提供高速互联网服务。",
         "SpaceX 成功发射新一批星链卫星，在轨总数再创新高"),
        ("华为今日发布了全新一代旗舰手机 Mate {number} 系列，搭载自研麒麟芯片，支持{feature}技术。余承东在发布会上表示，这款手机在摄影、续航和系统流畅度方面都有显著提升，起售价为{price}元。",
         "华为发布 Mate 系列旗舰手机，搭载自研麒麟芯片"),
        ("中科院某研究所团队宣布，他们在{field}领域取得重要进展，研发出一种新型{material}材料，可将{application}效率提升{percent}%。该技术有望在未来{year}年内投入实际应用。",
         "中科院研发新型材料，有望大幅提升应用效率"),

        # 体育
        ("在昨晚进行的某联赛第{round}轮比赛中，{teamA}主场{score}战胜{teamB}。{player}在比赛中独中{goals}元，成为全场最佳。赛后主教练表示，球队在防守端仍有提升空间，但进攻端的表现令人满意。",
         "{teamA}主场击败{teamB}，{player}独中{goals}元"),
        ("2024年巴黎奥运会{event}项目决赛结束，中国选手{name}以{score}的成绩夺得金牌，这也是中国代表团在本届奥运会的第{number}枚金牌。{name}赛后激动表示，这块金牌是对自己多年训练的最好回报。",
         "中国选手{name}夺得奥运{event}金牌"),
        ("NBA 季后赛西部决赛，{teamA}以{score}击败{teamB}，总比分{series}领先。{player}全场砍下{points}分{rebounds}篮板{assists}助攻的豪华数据，率领球队取得关键胜利。",
         "{teamA}击败{teamB}，{player}砍下豪华数据"),

        # 财经
        ("国家统计局今日发布数据显示，今年一季度国内生产总值同比增长{percent}%，高于市场预期。其中，{sector}行业表现尤为亮眼，增速达到{percent2}%。分析人士认为，随着各项稳经济政策落地，全年经济有望保持稳定增长。",
         "一季度 GDP 同比增长{percent}%，{sector}行业表现亮眼"),
        ("某知名电商企业今日发布季度财报，营收达{amount}亿元，同比增长{percent}%。但净利润同比下降{percent2}%，主要受{reason}影响。公司 CEO 表示，将持续加大{investment}投入，以应对市场竞争。",
         "某电商企业季度营收增长{percent}%，净利润承压"),
        ("央行今日宣布下调金融机构存款准备金率{percent}个百分点，释放长期资金约{amount}亿元。专家表示，此次降准旨在支持实体经济发展，降低融资成本，对股市和楼市均构成利好。",
         "央行降准{percent}个百分点，释放资金约{amount}亿元"),

        # 国际
        ("联合国安理会今日就{issue}召开紧急会议，{country}代表在会上呼吁各方保持克制，通过对话解决争端。中国代表重申了中方在{issue}问题上的立场，强调应尊重各国主权和领土完整。",
         "联合国安理会紧急讨论{issue}，中方呼吁对话解决"),
        ("{countryA}与{countryB}今日签署双边贸易协定，预计将使两国年度贸易额增长{percent}%。协定涵盖{sectors}等领域，双方同意逐步降低关税壁垒。",
         "{countryA}与{countryB}签署贸易协定，预计贸易额增长{percent}%"),
        ("某国今日宣布对{sector}产品加征{percent}%关税，引发国际市场震动。分析认为，此举可能引发新一轮贸易摩擦，对相关产业链造成冲击。",
         "某国宣布加征关税，引发国际市场震动"),

        # 社会
        ("教育部今日发布通知，要求各地中小学全面落实{policy}，确保学生每天{time}小时的{activity}时间。通知还强调，严禁学校以任何理由占用学生的{activity}时间。",
         "教育部要求落实{policy}，确保学生每天{time}小时{activity}时间"),
        ("某城市今日启动{project}民生工程，计划在未来{year}年内新建{number}所社区医院、{number2}个养老服务中心和{number3}个幼儿园。项目总投资约{amount}亿元。",
         "某城市启动民生工程，计划新建社区医院和养老中心"),
        ("全国铁路今日起实行新的列车运行图，新增开行旅客列车{number}对，主要覆盖{regions}等方向。同时，部分热门线路的运行时间进一步压缩，最快仅需{time}小时。",
         "全国铁路实行新运行图，新增列车{number}对"),

        # 文化
        ("第{number}届某国际电影节今日在{city}开幕，来自{countries}个国家和地区的{number2}部影片入围主竞赛单元。中国导演{name}的新作《{film}》将角逐金棕榈奖。",
         "第{number}届国际电影节开幕，中国影片入围主竞赛"),
        ("故宫博物院今日宣布，{exhibition}特展将于下月{date}日正式对外开放，展出文物{number}件，其中包括{treasure}等国宝级文物。展览将持续至{date2}日。",
         "故宫{exhibition}特展下月开幕，将展出国宝级文物"),
        ("著名作家{name}的新书《{book}》今日首发，上市首日销量突破{number}万册。该书以{theme}为主题，被评论界誉为{name}创作生涯的又一高峰。",
         "作家{name}新书首日销量破{number}万册"),
    ]

    rows = []
    vars_pool = {
        "number": lambda: random.randint(50, 500),
        "number2": lambda: random.randint(10, 100),
        "number3": lambda: random.randint(5, 50),
        "total": lambda: random.randint(3000, 8000),
        "times": lambda: random.randint(100, 10000),
        "year": lambda: random.randint(2025, 2030),
        "percent": lambda: random.randint(3, 15),
        "percent2": lambda: random.randint(5, 25),
        "price": lambda: random.choice([4999, 5999, 6999, 7999, 8999]),
        "amount": lambda: random.randint(100, 5000),
        "round": lambda: random.randint(15, 38),
        "score": lambda: f"{random.randint(1, 5)}:{random.randint(0, 3)}",
        "goals": lambda: random.randint(2, 4),
        "points": lambda: random.randint(25, 55),
        "rebounds": lambda: random.randint(5, 15),
        "assists": lambda: random.randint(3, 12),
        "series": lambda: random.choice(["2:1", "3:1", "3:2", "2:0"]),
        "event": lambda: random.choice(["跳水", "乒乓球", "羽毛球", "举重", "射击", "游泳", "体操"]),
        "name": lambda: random.choice(["张伟", "李娜", "王芳", "刘洋", "陈明", "赵强", "孙丽", "周杰", "吴刚", "郑浩"]),
        "player": lambda: random.choice(["詹姆斯", "库里", "杜兰特", "字母哥", "东契奇", "约基奇", "塔图姆", "巴特勒"]),
        "teamA": lambda: random.choice(["湖人", "勇士", "凯尔特人", "掘金", "热火", "太阳", "雄鹿", "快船"]),
        "teamB": lambda: random.choice(["火箭", "雷霆", "森林狼", "尼克斯", "76人", "独行侠", "国王", "鹈鹕"]),
        "field": lambda: random.choice(["新能源", "生物医药", "人工智能", "新材料", "量子信息"]),
        "material": lambda: random.choice(["钙钛矿", "石墨烯", "碳纳米管", "超导", "拓扑绝缘体"]),
        "application": lambda: random.choice(["太阳能电池", "储能电池", "催化剂", "传感器", "芯片散热"]),
        "feature": lambda: random.choice(["卫星通信", "卫星通话", "AI摄影", "折叠屏", "屏下指纹"]),
        "sector": lambda: random.choice(["高技术制造", "新能源", "信息技术", "生物医药", "消费服务"]),
        "reason": lambda: random.choice(["市场竞争加剧", "物流成本上升", "广告投入增加", "新业务发展"]),
        "investment": lambda: random.choice(["技术研发", "物流基础设施", "海外市场", "供应链优化"]),
        "issue": lambda: random.choice(["地区冲突", "人道主义危机", "网络安全", "气候变化", "贸易摩擦"]),
        "country": lambda: random.choice(["中国", "俄罗斯", "法国", "德国", "巴西", "印度", "日本", "韩国"]),
        "countryA": lambda: random.choice(["中国", "美国", "欧盟", "日本", "韩国", "东盟"]),
        "countryB": lambda: random.choice(["越南", "马来西亚", "印尼", "泰国", "菲律宾", "新加坡"]),
        "sectors": lambda: random.choice(["农产品、电子产品和纺织品", "汽车、机械和化工", "能源、矿产和基建", "服务贸易和数字产品"]),
        "policy": lambda: random.choice(["双减", "体教融合", "劳动教育", "心理健康教育"]),
        "time": lambda: random.choice(["1", "2", "1.5", "0.5"]),
        "activity": lambda: random.choice(["体育", "睡眠", "课外阅读", "社会实践"]),
        "project": lambda: random.choice(["十五分钟生活圈", "一老一小", "社区嵌入式服务", "城市更新"]),
        "regions": lambda: random.choice(["中西部、东北和西南", "长三角、珠三角和京津冀", "海南、云南和新疆", "沿海城市和省会城市"]),
        "city": lambda: random.choice(["戛纳", "柏林", "威尼斯", "东京", "上海", "北京", "釜山"]),
        "countries": lambda: random.randint(30, 90),
        "film": lambda: random.choice(["追光", "归途", "浮城", " silent", "破局", "新生"]),
        "exhibition": lambda: random.choice(["千里江山", "丝绸之路", "明清家具", "青铜器", "书画精品"]),
        "treasure": lambda: random.choice(["清明上河图", "千里江山图", "兰亭序", "四羊方尊", "金缕玉衣"]),
        "book": lambda: random.choice(["长夜", "归途", "浮世", "无声的告白", "边缘", "回声"]),
        "theme": lambda: random.choice(["都市人群的孤独", "时代变迁中的个体", "记忆与遗忘", "身份认同"]),
        "date": lambda: random.randint(1, 28),
        "date2": lambda: random.randint(1, 28),
    }

    for _ in range(count):
        article, summary = random.choice(articles)
        # 替换变量
        for var_name, gen in vars_pool.items():
            while '{' + var_name + '}' in article or '{' + var_name + '}' in summary:
                val = str(gen())
                article = article.replace('{' + var_name + '}', val, 1)
                summary = summary.replace('{' + var_name + '}', val, 1)
        rows.append((article, summary))

    return rows


# ==================== 智能问答数据 ====================

def generate_qa_data(count=3000):
    """生成智能问答训练数据"""

    qa_pairs = [
        # 历史
        ("中国历史上第一个统一的封建王朝是哪个？", "中国历史上第一个统一的封建王朝是秦朝，由秦始皇嬴政于公元前221年建立，结束了春秋战国以来长达五百多年的分裂局面。", "秦朝"),
        ("唐朝的开国皇帝是谁？", "唐朝的开国皇帝是李渊，即唐高祖。他于公元618年在长安称帝，建立唐朝。", "李渊"),
        ("四大发明包括哪些？", "中国古代四大发明包括造纸术、印刷术、火药和指南针。这四项发明对中国乃至世界文明的发展产生了深远影响。", "造纸术、印刷术、火药、指南针"),
        ("郑和下西洋发生在哪个朝代？", "郑和下西洋发生在明朝永乐年间，从1405年到1433年，郑和先后七次率领庞大的船队出使西洋。", "明朝"),
        ("辛亥革命发生于哪一年？", "辛亥革命发生于1911年，这次革命推翻了清朝统治，结束了中国两千多年的封建帝制。", "1911年"),

        # 地理
        ("世界上最大的沙漠是哪个？", "世界上最大的沙漠是撒哈拉沙漠，位于非洲北部，面积约906万平方公里，相当于整个中国的面积。", "撒哈拉沙漠"),
        ("长江的全长约多少公里？", "长江是中国第一大河，也是亚洲第一长河，全长约6300公里，流经青海、西藏、四川、云南、重庆等11个省市区。", "约6300公里"),
        ("世界上最深的海沟是什么？", "世界上最深的海沟是马里亚纳海沟，位于西太平洋，最深处约11034米，被称为挑战者深渊。", "马里亚纳海沟"),
        ("北极和南极哪个更冷？", "南极更冷。南极大陆的平均气温约为零下49摄氏度，而北极地区的平均气温约为零下18摄氏度。", "南极"),
        ("中国最大的岛屿是哪个？", "中国最大的岛屿是台湾岛，面积约3.6万平方公里。其次是海南岛，面积约3.4万平方公里。", "台湾岛"),

        # 科技
        ("光在真空中的传播速度是多少？", "光在真空中的传播速度约为每秒30万公里，即299,792,458米/秒，这是自然界中的极限速度。", "约每秒30万公里"),
        ("DNA的中文全称是什么？", "DNA的中文全称是脱氧核糖核酸，它是生物体内存储遗传信息的重要分子，由两条互补的核苷酸链组成双螺旋结构。", "脱氧核糖核酸"),
        ("计算机中的CPU是什么的缩写？", "CPU是Central Processing Unit的缩写，中文称为中央处理器，是计算机的核心部件，负责执行指令和处理数据。", "中央处理器"),
        ("什么是温室效应？", "温室效应是指大气中的温室气体（如二氧化碳、甲烷等）吸收地面辐射的热量，使地球表面温度升高的现象。适度的温室效应使地球保持适宜温度，但过度增强会导致全球变暖。", "大气中温室气体吸收热量导致温度升高的现象"),
        ("WiFi使用的是什么类型的电磁波？", "WiFi使用的是无线电波中的微波频段，常见的2.4GHz和5GHz都属于微波范围，属于非电离辐射。", "微波/无线电波"),

        # 文学
        ("《三国演义》的作者是谁？", "《三国演义》的作者是元末明初小说家罗贯中，是中国第一部长篇章回体历史演义小说，描写了从东汉末年到西晋初年近百年的历史风云。", "罗贯中"),
        ("鲁迅的原名是什么？", "鲁迅的原名是周树人，字豫才，浙江绍兴人。他是中国现代文学的奠基人之一，代表作有《狂人日记》《阿Q正传》等。", "周树人"),
        ("《诗经》分为哪三个部分？", "《诗经》分为风、雅、颂三个部分。'风'是各地民歌，'雅'是宫廷乐歌，'颂'是宗庙祭祀乐歌，共收录诗歌305篇。", "风、雅、颂"),
        ("唐宋八大家包括哪些人？", "唐宋八大家包括唐代的韩愈、柳宗元和宋代的欧阳修、苏洵、苏轼、苏辙、王安石、曾巩。他们是唐宋时期古文运动的代表人物。", "韩愈、柳宗元、欧阳修、苏洵、苏轼、苏辙、王安石、曾巩"),
        ("《西游记》中唐僧取经的目的地是哪里？", "《西游记》中唐僧取经的目的地是天竺国的大雷音寺，即今天的印度地区，去取的是大乘佛教真经。", "天竺国大雷音寺（印度）"),

        # 艺术
        ("《蒙娜丽莎》的作者是谁？", "《蒙娜丽莎》的作者是意大利文艺复兴时期画家列奥纳多·达·芬奇，这幅画作于1503年至1519年间，现藏于法国巴黎卢浮宫。", "达·芬奇"),
        ("贝多芬是哪国作曲家？", "贝多芬是德国作曲家，维也纳古典乐派代表人物之一。他在耳聋的情况下创作了大量不朽作品，如《命运交响曲》《月光奏鸣曲》等。", "德国"),
        ("京剧的四大行当是什么？", "京剧的四大行当是生、旦、净、丑。'生'指男性角色，'旦'指女性角色，'净'指花脸角色，'丑'指喜剧角色。", "生、旦、净、丑"),
        ("芭蕾舞起源于哪个国家？", "芭蕾舞起源于文艺复兴时期的意大利，后在法国宫廷得到发展，最终在俄罗斯达到艺术巅峰。", "意大利"),
        ("中国传统建筑中，屋顶的飞檐设计有什么作用？", "飞檐设计有多种作用：一是扩大采光面，利于雨水排泄；二是增加建筑的美感，使屋顶线条更加灵动；三是等级象征，不同形式的飞檐代表不同的建筑等级。", "扩大采光、利于排水、增加美感、等级象征"),

        # 生活常识
        ("人体最大的器官是什么？", "人体最大的器官是皮肤。成人皮肤总面积约1.5至2平方米，重量约占体重的16%，具有保护、调节体温、感觉等多种功能。", "皮肤"),
        ("为什么天空是蓝色的？", "天空呈现蓝色是因为瑞利散射。太阳光中的蓝光波长较短，更容易被大气分子散射到各个方向，所以我们看到的天空是蓝色的。", "瑞利散射使蓝光被散射"),
        ("一天喝多少水比较合适？", "成年人每天建议饮水1500至2000毫升，约7至8杯。具体需求量因个人体质、活动量和气候条件而异，运动或炎热天气应适当增加。", "1500至2000毫升"),
        ("为什么冰箱门很难打开？", "冰箱门难打开是因为门封条内的空气被冷却后形成低压区，外部大气压将门压紧。等内外气压平衡后，门就容易打开了。", "内外气压差导致"),
        ("彩虹为什么是弯的？", "彩虹是弯的是因为光线在水滴中发生折射、反射和色散后，只有与入射光成约42度角的方向才能被观察到，这些方向在天空中形成一个圆锥面，我们看到的是这个圆锥与地面相交的弧形。", "光线在水滴中形成42度圆锥面"),
    ]

    # 扩展生成：对基础QA对进行同义改写
    def paraphrase_question(question, context, answer):
        """对问题进行同义改写"""
        patterns = [
            lambda q, c, a: (f"请问{q}", c, a),
            lambda q, c, a: (f"我想知道{q}", c, a),
            lambda q, c, a: (f"能告诉我{q}吗？", c, a),
            lambda q, c, a: (q.replace("哪个", "什么") if "哪个" in q else q, c, a),
            lambda q, c, a: (q.replace("是什么", "指的是什么") if "是什么" in q else q, c, a),
            lambda q, c, a: (f"关于{q[:-1] if q.endswith('？') else q}，你了解吗？", c, a),
        ]
        if random.random() < 0.3:
            f = random.choice(patterns)
            return f(question, context, answer)
        return question, context, answer

    rows = []
    base_count = len(qa_pairs)
    # 每个基础QA对生成多个变体
    repeats = count // base_count + 1

    for q, c, a in qa_pairs:
        for _ in range(repeats):
            if len(rows) >= count:
                break
            q2, c2, a2 = paraphrase_question(q, c, a)
            rows.append((q2, c2, a2))

    random.shuffle(rows)
    return rows[:count]


if __name__ == "__main__":
    print("=" * 50)
    print("开始生成多样化训练数据")
    print("=" * 50)

    sentiment_rows = generate_sentiment_data(3000)
    save_csv("data/sentiment.csv", ["text", "label"], sentiment_rows)

    summary_rows = generate_summary_data(3000)
    save_csv("data/summary.csv", ["text", "summary"], summary_rows)

    qa_rows = generate_qa_data(3000)
    save_csv("data/qa.csv", ["question", "context", "answer"], qa_rows)

    print("=" * 50)
    print("数据生成完成！")
    print("=" * 50)
