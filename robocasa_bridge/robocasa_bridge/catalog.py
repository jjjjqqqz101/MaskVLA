"""Read-only official object catalog. No simulator or CUDA import is required."""
import ast,hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .paths import ROOT, ASSETS, ROBOCASA_ROOT
REGISTRY = ROBOCASA_ROOT / "robocasa/models/objects/kitchen_objects.py"
LEGACY_CERAMIC_PATH_SHA256 = "c1c33b90c94f9662f87779baf4e90e89f47047ef5de532c21304f4b061599505"
SHARED_CERAMIC="shared/textures/ceramic.png"
CERAMIC_SHA256="beef464a2aa9143ba6a25a44adcbfd1f1aff16c5969edbf08019b61025388402"

MAX_OBJECTS=6
MAX_SIZE=np.array([.13,.13,.18])
MIN_FIT=.45
ZH={"apple":"苹果","avocado":"牛油果","banana":"香蕉","orange":"橙子","lemon":"柠檬","lime":"青柠",
    "peach":"桃子","pear":"梨","plum":"李子","strawberry":"草莓","grapes":"葡萄","watermelon":"西瓜",
    "pineapple":"菠萝","mango":"芒果","kiwi":"猕猴桃","carrot":"胡萝卜","potato":"土豆","sweet_potato":"红薯",
    "tomato":"番茄","onion":"洋葱","cucumber":"黄瓜","eggplant":"茄子","bell_pepper":"甜椒","broccoli":"西兰花",
    "corn":"玉米","egg":"鸡蛋","cheese":"奶酪","bread":"面包","bagel":"贝果","baguette":"法棍","croissant":"可颂",
    "donut":"甜甜圈","cookie":"曲奇","cake":"蛋糕","chocolate":"巧克力","steak":"牛排","fish":"鱼",
    "chicken_drumstick":"鸡腿","sausage":"香肠","bowl":"碗","plate":"盘子","cup":"杯子","mug":"马克杯",
    "glass_cup":"玻璃杯","honey_bottle":"蜂蜜瓶","lemon_wedge":"柠檬角","cream_cheese_stick":"奶油奶酪棒","milk_carton":"牛奶盒","glass":"玻璃杯","can":"易拉罐","beer":"啤酒","bottled_water":"瓶装水","bottled_drink":"瓶装饮料",
    "boxed_drink":"盒装饮料","milk":"牛奶","juice":"果汁","cereal":"谷物盒","bar":"能量棒","bar_soap":"香皂",
    "soap_dispenser":"洗手液瓶","sponge":"海绵","fork":"叉子","knife":"刀","spoon":"勺子","spatula":"锅铲",
    "ladle":"汤勺","tongs":"夹子","pan":"平底锅","pot":"锅","saucepan":"炖锅","kettle_non_electric":"水壶",
    "tray":"托盘","cutting_board":"砧板","salt":"盐","pepper":"胡椒","sugar":"糖","stool":"凳子","plant":"植物",
    "dish_rack":"碗碟架","knife_block":"刀架","paper_towel_holder":"纸巾架","utensil_rack":"厨具架","utensil_set":"厨具套装"}
ZH.update(dict(line.split("=",1) for line in '''bagged_food=袋装食品
boxed_food=盒装食品
bread_flat=薄饼
candle=蜡烛
canned_food=罐头食品
chips=薯片
coffee_cup=咖啡杯
condiment_bottle=调料瓶
cupcake=纸杯蛋糕
garlic=大蒜
hot_dog=热狗
jam=果酱
jug=大水壶
jug_wide_opening=广口水壶
ketchup=番茄酱
liquor=烈酒瓶
mushroom=蘑菇
rolling_pin=擀面杖
scissors=剪刀
shaker=调料罐
spray=喷雾瓶
squash=南瓜
tangerine=橘子
teapot=茶壶
waffle=华夫饼
water_bottle=水瓶
wine=葡萄酒
yogurt=酸奶
dates=椰枣
lemonade=柠檬饮料
walnut=核桃
cheese_grater=奶酪刨丝器
syrup_bottle=糖浆瓶
scallops=扇贝
candy=糖果
whisk=打蛋器
pitcher=带柄水壶
ice_cream=冰淇淋
cherry=樱桃
peanut_butter=花生酱
thermos=保温瓶
ham=火腿
dumpling=饺子
cabbage=卷心菜
lettuce=生菜
ginger=生姜
ice_cube_tray=冰格
shrimp=虾
cantaloupe=哈密瓜
spaghetti_box=意大利面盒
chili_pepper=辣椒
celery=芹菜
burrito=墨西哥卷饼
olive_oil_bottle=橄榄油瓶
kebabs=烤肉串
bottle_opener=开瓶器
chicken_breast=鸡胸肉
jello_cup=果冻杯
lobster=龙虾
brussel_sprout=抱子甘蓝
sushi=寿司
baking_sheet=烘焙烤盘
wine_glass=高脚酒杯
asparagus=芦笋
lamb_chop=羊排
pickle=腌黄瓜
bacon=培根
canola_oil=菜籽油
pizza_cutter=披萨滚刀
pomegranate=石榴
apricot=杏
beet=甜菜
radish=萝卜
salsa=莎莎酱
artichoke=洋蓟
scone=司康饼
hamburger=汉堡
raspberry=树莓
tacos=墨西哥夹饼
vinegar=醋
zucchini=西葫芦
pork_loin=猪里脊
pork_chop=猪排
coconut=椰子
cauliflower=花菜
lollipop=棒棒糖
salami=萨拉米香肠
butter_stick=黄油块
can_opener=开罐器
tofu=豆腐
skewers=串签
colander=沥水篮
reamer=柑橘榨汁器
ice_cube=冰块
jar=储物罐
measuring_cup=量杯
sandwich_bread=吐司面包
strainer=滤网
sugar_cube=方糖
cinnamon=肉桂
cookie_dough_ball=曲奇面团球
digital_scale=电子秤
flour_bag=面粉袋
hotdog_bun=热狗面包
mayonnaise=蛋黄酱
mustard=芥末酱
oven_tray=烤箱托盘
paprika=红椒粉
turmeric=姜黄粉
pickle_slice=腌黄瓜片
tomato_slice=番茄片
turkey_slice=火鸡肉片
straw=吸管
tupperware=保鲜盒
marshmallow=棉花糖
oil_and_vinegar_bottle=油醋瓶
pancake=松饼
peeler=削皮器
saucepan_with_lid=带盖炖锅
saucepan_lid=炖锅盖
wooden_spoon=木勺
aluminum_foil=铝箔
dish_brush=洗碗刷
pizza=披萨
kebab_skewer=烧烤签
salt_and_pepper_shaker=盐胡椒调料瓶
basket=篮子
placemat=餐垫
blender_jug=搅拌机杯'''.splitlines()))
_CACHE=None
_CACHE_TIME=0.
_REGISTRY_CACHE=None
def _literal(node):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,(ast.Tuple,ast.List)):return [_literal(x) for x in node.elts]
    if isinstance(node,ast.Dict):return {_literal(k):_literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -_literal(node.operand)
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="dict" and not node.args:
        return {k.arg:_literal(k.value) for k in node.keywords}
    raise ValueError("Unsupported official object-registry expression")
def registry():
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        tree=ast.parse(REGISTRY.read_text())
        assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=="OBJ_CATEGORIES" for t in x.targets))
        # Evaluate only the trusted official registry expression; do not import
        # its module (which eagerly snapshots the asset tree during downloads).
        _REGISTRY_CACHE=eval(compile(ast.Expression(assignment.value),str(REGISTRY),"eval"),
            {"__builtins__":{"dict":dict,"set":set,"range":range,"list":list}})
    return _REGISTRY_CACHE
def asset_path(path):
    p=Path(path).resolve();root=ASSETS.resolve()
    if p!=root and root not in p.parents:raise ValueError("Object path leaves the RoboCasa asset root")
    return p
def is_legacy_ceramic(reference):
    return isinstance(reference, str) and hashlib.sha256(reference.encode()).hexdigest() == LEGACY_CERAMIC_PATH_SHA256


def reference_path(model_path,file,tag,compiler=None):
    """Resolve asset references without accepting arbitrary outside paths.

    The sole legacy exception is the exact published Aigen ceramic texture.
    It resolves to a SHA-pinned local official copy, never the author's path.
    """
    if is_legacy_ceramic(file):
        if tag!="texture":raise ValueError("Legacy ceramic relocation is permitted only for a texture")
        target=asset_path(ASSETS/SHARED_CERAMIC)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=CERAMIC_SHA256:
            raise ValueError("Official shared ceramic texture is missing or its checksum differs")
        return target
    base=Path(model_path).parent
    if compiler is not None and tag in ("mesh","texture"):
        base=base/compiler.get("meshdir" if tag=="mesh" else "texturedir","")
    return asset_path(base/file)

def reference_relocations(root):
    for node in root.iter():
        reference = node.get("file")
        if not is_legacy_ceramic(reference):
            continue
        return [dict(original_reference=reference,local_asset_path=SHARED_CERAMIC,
            sha256=CERAMIC_SHA256,provenance="official robosuite textures/ceramic.png; exact Aigen author-path relocation; source XML retained")]
    return []

def _models(category,source,properties):
    disk="aigen_objs" if source=="aigen" else source
    folders=properties.get("model_folders") or [disk+"/"+category]
    result=[]
    for folder in folders:
        base=asset_path(ASSETS/"objects"/folder)
        if not base.is_dir():continue
        pattern="*/*/model.xml" if properties.get("is_auxiliary_obj") else "*/model.xml"
        for path in sorted(base.glob(pattern)):
            if path.parent.name in properties.get("exclude",[]):continue
            if path.is_symlink() or not path.is_file():continue
            relative=path.relative_to(ASSETS)
            result.append(dict(instance_id=hashlib.sha256(str(relative).encode()).hexdigest()[:20],label=path.parent.name,path=path))
    return result
def catalog_index(refresh=False):
    global _CACHE,_CACHE_TIME
    if _CACHE is not None and not refresh and time.monotonic()-_CACHE_TIME<30:return _CACHE
    entries={}
    for cat,values in registry().items():
        for source in ("objaverse","aigen","lightwheel"):
            if source not in values:continue
            props={**values[source]}
            for key in ("is_auxiliary_obj","graspable","types"):props[key]=values.get(key,False)
            models=_models(cat,source,props);graspable=bool(values.get("graspable",False))
            reason=None if models and graspable and not values.get("is_auxiliary_obj") else (
                "资产尚未下载完成" if not models else "辅助部件不可单独生成" if values.get("is_auxiliary_obj") else
                "官方未标为可抓取物品，仅作为场景/资产展示")
            types=values.get("types",[])
            if isinstance(types,str):types=[types]
            cid=source+":"+cat
            entries[cid]=dict(catalog_id=cid,name=cat,name_zh=ZH.get(cat,cat.replace("_"," ")),source=source,
                graspable=graspable,types=list(types),instance_count=len(models),available=bool(models),
                selectable=reason is None,disabled_reason=reason,_instances=models,_properties=props)
    # The installed kitchen decor also belongs to the asset inventory, but
    # is not promoted into graspable categories absent from the registry.
    object_root=ASSETS/"objects"
    if object_root.is_dir():
        for library in sorted(object_root.iterdir()):
            if not library.is_dir():continue
            source="aigen" if library.name=="aigen_objs" else library.name
            if source not in ("objaverse","aigen","lightwheel"):continue
            for directory in sorted(library.iterdir()):
                if not directory.is_dir():continue
                cid=source+":"+directory.name
                if cid in entries:continue
                models=_models(directory.name,source,{})
                if not models:continue
                entries[cid]=dict(catalog_id=cid,name=directory.name,
                    name_zh=ZH.get(directory.name,directory.name.replace("_"," ")),source=source,
                    graspable=False,types=["decoration"],instance_count=len(models),available=True,
                    selectable=False,disabled_reason="厨房布景/未注册为可抓取目标，当前不支持台面采集",
                    _instances=models,_properties={})
    _CACHE=entries;_CACHE_TIME=time.monotonic()
    return entries
def _public(entry):return {k:v for k,v in entry.items() if not k.startswith("_")}
def instance_info(entry,item,check_references=False):
    result={k:v for k,v in item.items() if k!="path"}
    result.update(selectable=False,disabled_reason=entry["disabled_reason"])
    if not entry["selectable"]:return result
    try:
        path=asset_path(item["path"]);root=ET.parse(path).getroot()
        box=next((g for g in root.iter("geom") if g.get("name")=="reg_bbox"),None)
        if box is None:raise ValueError("模型缺少官方尺寸边界")
        half=np.fromstring(box.get("size",""),sep=" ")
        if box.get("type")=="cylinder" and len(half)==2:half=np.array([half[0],half[0],half[1]])
        if len(half)!=3 or not np.isfinite(half).all() or (half<=0).any():raise ValueError("模型尺寸无效")
        scale=np.asarray(entry["_properties"].get("scale",1.),float)
        if scale.ndim==0:scale=np.repeat(scale,3)
        native=half*2*scale;factor=float(min(1.,np.min(MAX_SIZE/native)))
        if factor<MIN_FIT:raise ValueError("物品过大，不适合当前固定台面和夹爪")
        if min(native*factor)<.004:raise ValueError("物品过薄/过小，不适合当前采集任务")
        if check_references:
            compiler=root.find("compiler")
            for elem in root.iter():
                file=elem.get("file")
                if not file:continue
                ref=reference_path(path,file,elem.tag,compiler)
                if not ref.is_file():raise ValueError("模型网格或纹理仍在下载")
        result.update(selectable=True,disabled_reason=None,bbox_size_m=(native*factor).tolist(),
            original_bbox_size_m=native.tolist(),applied_scale_factor=factor,scale=(scale*factor).tolist(),
            asset_reference_relocations=reference_relocations(root))
    except (ValueError,OSError,ET.ParseError) as exc:result["disabled_reason"]=str(exc)
    return result
def get_catalog(query="",source="",category_id="",offset=0,limit=30,refresh=False):
    entries=catalog_index(refresh);offset=max(0,int(offset));limit=max(1,min(100,int(limit)))
    q=str(query).strip().casefold()
    rows=[e for e in entries.values() if (not source or e["source"]==source) and
        (not q or q in (e["name"]+" "+e["name_zh"]+" "+" ".join(e["types"])).casefold())]
    rows.sort(key=lambda e:(not e["selectable"],e["name"],e["source"]))
    # Official categories can alias the same model directories. Keep the
    # download count distinct from category-instance aliases in the index.
    indexed_instances=len({str(item["path"]) for e in entries.values() for item in e["_instances"]})
    downloaded_instances=indexed_instances;library_status="indexed_available"
    try:
        manifest=json.loads((ASSETS/"object_asset_manifest.json").read_text())
        if (manifest.get("schema")=="mask_policy.robocasa.object_assets.v1" and
            manifest.get("status")=="complete" and manifest.get("model_count")==len(manifest.get("models",[]))):
            downloaded_instances=int(manifest["model_count"]);library_status="complete"
    except (OSError,ValueError,TypeError):pass
    result=dict(ok=True,categories=[_public(e) for e in rows[offset:offset+limit]],total=len(rows),
        offset=offset,limit=limit,sources=["objaverse","aigen","lightwheel"],max_objects=MAX_OBJECTS,
        downloaded_instances=downloaded_instances,indexed_unique_instances=indexed_instances,asset_library_status=library_status,
        selectable_categories=sum(bool(e["selectable"]) for e in entries.values()),
        library_notice="仅已完整下载且适合台面的官方物品可生成；可抓取标记不代表已有策略或脚本能抓取",
        sizing=dict(max_bbox_m=MAX_SIZE.tolist(),min_uniform_fit=MIN_FIT,scale_recorded=True))
    if category_id:
        if category_id not in entries:raise ValueError("Unknown object category")
        entry=entries[category_id]
        result.update(category=_public(entry),instances=[instance_info(entry,x) for x in entry["_instances"][offset:offset+limit]],
            instance_total=len(entry["_instances"]))
    return result
def resolve_selection(selection,seed):
    if not isinstance(selection,list) or not selection:raise ValueError("Select 1 to 6 tabletop objects")
    count=0
    for item in selection:
        if not isinstance(item,dict):raise ValueError("Each object selection must be an object")
        quantity=item.get("quantity",1)
        if type(quantity) is not int or not 1<=quantity<=MAX_OBJECTS:raise ValueError("quantity must be 1..6")
        count+=quantity
    if not 1<=count<=MAX_OBJECTS:raise ValueError("Select a total of 1 to 6 objects")
    rng=np.random.default_rng(seed);entries=catalog_index(refresh=True);chosen=[]
    for request in selection:
        cid=request.get("catalog_id")
        if cid not in entries:raise ValueError("Unknown object catalog_id")
        entry=entries[cid]
        if not entry["selectable"]:raise ValueError(entry["disabled_reason"])
        candidates=entry["_instances"];requested=request.get("instance_id")
        if requested is not None:
            candidates=[x for x in candidates if x["instance_id"]==requested]
            if not candidates:raise ValueError("Unknown instance_id for selected category")
        for _ in range(request.get("quantity",1)):
            selected=None;reasons=[]
            for index in rng.permutation(len(candidates)):
                item=candidates[int(index)];info=instance_info(entry,item,True)
                if info["selectable"]:selected=(item,info);break
                reasons.append(info["disabled_reason"])
            if selected is None:raise ValueError(entry["name"]+": no suitable fully downloaded instance ("+"; ".join(sorted(set(reasons)))+")")
            item,info=selected
            chosen.append(dict(id=f"item_{len(chosen)+1:02d}_{entry['name']}",catalog_id=cid,
                name=entry["name"],name_zh=entry["name_zh"],source=entry["source"],instance_id=info["instance_id"],
                instance_label=info["label"],model_path=str(item["path"].relative_to(ASSETS)),
                model_sha256=hashlib.sha256(item["path"].read_bytes()).hexdigest(),bbox_size_m=info["bbox_size_m"],
                original_bbox_size_m=info["original_bbox_size_m"],scale=info["scale"],
                applied_scale_factor=info["applied_scale_factor"],official_graspable=entry["graspable"],
                asset_reference_relocations=info["asset_reference_relocations"]))
    return chosen
def validate_resolved(objects):
    if not isinstance(objects,list) or not 1<=len(objects)<=MAX_OBJECTS:raise ValueError("Invalid recorded object selection")
    ids=[]
    for item in objects:
        path=asset_path(ASSETS/item["model_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item["model_sha256"]:
            raise ValueError("Recorded official object is missing or its XML changed")
        if not isinstance(item["id"],str) or not item["id"].startswith("item_"):raise ValueError("Invalid recorded object identity")
        scale=np.asarray(item["scale"],float)
        if scale.shape!=(3,) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError("Invalid recorded object scale")
        ids.append(item["id"])
    if len(set(ids))!=len(ids):raise ValueError("Recorded object IDs must be unique")
    return objects
def build_object(info):
    """Use official MJCFObject implementation with scratch XML inside PROJECT."""
    import tempfile
    from robocasa.models.objects.objects import MJCFObject
    entry=catalog_index().get(info["catalog_id"])
    if entry is None:raise ValueError("Recorded category is absent from official registry")
    original=asset_path(ASSETS/info["model_path"]);root=ET.parse(original).getroot();compiler=root.find("compiler")
    for elem in root.iter():
        file=elem.get("file")
        if not file:continue
        elem.set("file",str(reference_path(original,file,elem.tag,compiler)))
    if compiler is not None:
        compiler.attrib.pop("meshdir",None);compiler.attrib.pop("texturedir",None)
    scratch=ROOT/"work/robocasa_bridge/object_xml";scratch.mkdir(parents=True,exist_ok=True)
    props=entry["_properties"]
    kwargs={k:props[k] for k in ("solimp","solref","density","friction","priority") if k in props}
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path=Path(directory)/"model.xml";ET.ElementTree(root).write(path,encoding="unicode")
        obj=MJCFObject(name=info["id"],mjcf_path=str(path),scale=info["scale"],**kwargs)
    return obj
