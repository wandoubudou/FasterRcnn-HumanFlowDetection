from lib.config.config import FLAGS2

people_num = {"time":[],"number":[],"place":FLAGS2['cur_place']}
processed_people_num = {'time':[],'number':[],'place':FLAGS2['cur_place']}

def init_people_num():
    people_num1 = {"time":[],"number":[],"place":FLAGS2['cur_place']}
    return people_num1
def set_place(place):
    people_num['place'] = place
    processed_people_num['place'] = place
    FLAGS2['cur_place']=place
    return True

def statistic_num(time,number):     #若time 和number 都不为空则将其载入people_num中，否则返回失败，跳过本次统计
    if time!=None and number!=None:
        people_num['time'].append(time)
        people_num['number'].append(number)
        return True
    else:
        return False

def get_counted_people():
    return people_num

def process_json_info(cur_FPS):
    total_people_num = 0
    processed_people_num = {'time': [], 'number': [], 'place': FLAGS2['cur_place']}
    times = 0
    for i in range(len(people_num['time'])):
        times += 1
        total_people_num += people_num['number'][i]
        if times>=cur_FPS:
            ave_number = int((total_people_num/times)+0.5)
            processed_people_num['number'].append(ave_number)
            times = 0
            total_people_num = 0
    people_num1 = init_people_num()
    people_num['time'] = people_num1['time']
    people_num['place'] = people_num1['place']
    people_num['number'] = people_num1['number']
    return processed_people_num