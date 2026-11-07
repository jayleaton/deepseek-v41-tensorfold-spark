import json,pathlib,sys,math,statistics,random,collections
H=pathlib.Path.home();O=pathlib.Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True);rows=json.loads((H/'work/imr-t5-s1/transport/ab3-pairs.json').read_text());families=sorted({r['family'] for r in rows});tasks={f:sorted({r['id'] for r in rows if r['family']==f}) for f in families};data={(r['id'],r['seed']):r for r in rows};arms=['plain','lane','full','only'];rng=random.Random(20261104);contrasts=[('full','lane'),('full','plain'),('lane','plain'),('only','plain'),('only','lane'),('full','only')];boot={a+'/'+b:[] for a,b in contrasts};solves=[];family_effect={f:[] for f in families};F=sum(r['full']['T'] for r in rows);L=sum(r['lane']['T'] for r in rows);R=F/L
for draw in range(20000):
 sums={a:0 for a in arms};sf=sl=0
 for family in families:
  ff=ll=0
  for _ in tasks[family]:
   task=rng.choice(tasks[family])
   for _ in range(3):
    r=data[task,rng.choice([101,102,103])]
    for a in arms:sums[a]+=r[a]['T']
    ff+=r['full']['T'];ll+=r['lane']['T'];sf+=r['full']['solved'];sl+=r['lane']['solved']
  family_effect[family].append((ff-R*ll)/L)
 for a,b in contrasts:boot[a+'/'+b].append(sums[a]/sums[b])
 solves.append((sf-sl)/180)
 if draw%5000==0:print('draw',draw,flush=True)
result={'contrasts':{},'solves':{},'by_seed':{},'by_family':{},'planning':{}}
for a,b in contrasts:
 v=sorted(boot[a+'/'+b]);result['contrasts'][a+'/'+b]={'change_pct':100*(sum(r[a]['T'] for r in rows)/sum(r[b]['T'] for r in rows)-1),'CI95_pct':[100*(v[500]-1),100*(v[19499]-1)]}
loss=[r for r in rows if r['lane']['solved'] and not r['full']['solved']];gain=[r for r in rows if r['full']['solved'] and not r['lane']['solved']];solves.sort();result['solves']={'losses':[(r['id'],r['seed']) for r in loss],'gains':[(r['id'],r['seed']) for r in gain],'repeat_tasks':[t for t,n in collections.Counter(r['id'] for r in loss).items() if n>=2],'diff_CI95':[solves[500],solves[19499]],'pass':len(loss)<=3 and len(gain)>=len(loss) and all(n<2 for n in collections.Counter(r['id'] for r in loss).values())}
for s in [101,102,103]:
 z=[r for r in rows if r['seed']==s];result['by_seed'][s]=100*(sum(r['full']['T'] for r in z)/sum(r['lane']['T'] for r in z)-1)
for f in families:
 z=[r for r in rows if r['family']==f];result['by_family'][f]={'full':sum(r['full']['T'] for r in z),'lane':sum(r['lane']['T'] for r in z),'linear_boot_variance':statistics.variance(family_effect[f])}
vsum=sum(z['linear_boot_variance'] for z in result['by_family'].values())
for f in families:result['by_family'][f]['variance_share']=result['by_family'][f]['linear_boot_variance']/vsum
changes={t:100*(sum(r['full']['T'] for r in rows if r['id']==t)/sum(r['lane']['T'] for r in rows if r['id']==t)-1) for t in {r['id'] for r in rows}};result['median_task_change']=statistics.median(changes.values());result['leave_one_task_out']={t:100*((F-sum(r['full']['T'] for r in rows if r['id']==t))/(L-sum(r['lane']['T'] for r in rows if r['id']==t))-1) for t in changes}
logsd=statistics.stdev(map(math.log,boot['full/lane']));z=statistics.NormalDist();gap=math.log(.95/R);result['planning']={'R':R,'log_ratio_boot_sd':logsd,'log_CI_halfwidth':(math.log(sorted(boot['full/lane'])[19499])-math.log(sorted(boot['full/lane'])[500]))/2,'gate_gap':gap,'assumption':'same true ratio/variance and three paired seeds; approximate primary-token power only, independent new task clusters','N80_one_contrast':math.ceil(60*((z.inv_cdf(.975)+z.inv_cdf(.80))*logsd/gap)**2),'N90_one_contrast':math.ceil(60*((z.inv_cdf(.975)+z.inv_cdf(.90))*logsd/gap)**2),'N80_three_Bonferroni_contrasts_proxy':math.ceil(60*((z.inv_cdf(1-.05/6)+z.inv_cdf(.8))*logsd/gap)**2)}
for a in ['lane','full','only']:
 z=[r for r in rows if r['plain']['censored'] or r[a]['censored']];savings=sum(r['plain']['T']-r[a]['T'] for r in rows);result.setdefault('plain_censor_sensitivity',{})[a]={'censored_pairs':len(z),'net_contribution_pct':100*sum(r['plain']['T']-r[a]['T'] for r in z)/savings,'complete_both_change_pct':100*(sum(r[a]['T'] for r in rows if r not in z)/sum(r['plain']['T'] for r in rows if r not in z)-1)}
for a in arms:
 result.setdefault('censoring',{})[a]={'count':sum(r[a]['censored'] for r in rows),'cuts':sum(r[a]['cut'] for r in rows)}
(O/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(O/'script.py').write_text(pathlib.Path(__file__).read_text());print(json.dumps({k:v for k,v in result.items() if k!='leave_one_task_out'},indent=2))
