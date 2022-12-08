import numpy as np
import sys
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont


#Глобальные параметры (Россия и мир):
#Технические:
img = Image.open('Source.png').convert('RGB')
img_blue_colors = [0 for i in range(0, 256)]


def earth_distance(s_lat, s_lng, e_lat, e_lng):
   r = 6373.0
   s_lat = s_lat*np.pi/180.0
   s_lng = np.deg2rad(s_lng)
   e_lat = np.deg2rad(e_lat)
   e_lng = np.deg2rad(e_lng)
   d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
   return 2 * r * np.arcsin(np.sqrt(d))


for i in range(0, 1770):
    for j in range(38, img.height):
        t = img.getpixel((i, j))
        if t[0] == 0 and t[1] == 0:
            x = i
            y = j - 38
            mod_s = (earth_distance(18.938 + 0.107 * i, 80.76 - 0.08 * j,
                                     18.938 + 0.107 * (i + 1), 80.76 - 0.08 * j) *
                      earth_distance(18.938 + 0.107 * i, 80.76 - 0.08 * j,
                                     18.938 + 0.107 * i, 80.76 - 0.08 * (j + 1))
                      )
            img_blue_colors[t[2]] += mod_s
gif_russians = []
gif_density = []
gif_gdp = []
gif_towners = []
assim_data = pd.read_excel('NationAssimTable.xlsx', index_col='Index')
img_num = {
    'nations': 0,
    'infra': 0,
    'poverty': 0,
    'literacy': 0,
    'gdp': 0,
    'urban': 0
} #Номер изображения
#Мир:
year = 1897
month = 1
global_max_med = 10.0 #Максимальный уровень медицины в мире
global_gdp_med = 400.0 #Средний ВВП на душу населения в развитых странах
global_stability = 1.0 #Мировая стабильность
global_openness = 1.0 #Мировая открытость
global_building_technology = 1.0 #Мировые строительные технологии
global_agro_technology = 1.0 #Мировые с/х технологии
#Россия:
russian_toleracy = 0.2 #Терпимость к меньшинствам
russian_openness = 1.0 #Открытость границ
russian_stability = 0.8 #Стабильность в стране
russian_democracy = 0.0 #Демократичность режима
russian_selfcon = 0.6 #Национальное самосознание русских
russian_humilation = 0.1 #Степень унижения русских/России
russian_unlabour = 90 #Возраст выхода на пенсию
russian_child = 12 #Возраст, с которого разрешён труд
russian_basic_cost = 5.0 #Базовая стоимость необходимых для жизни продуктов
russian_monthly_inflation = 1.001 #Месячная инфляция в России
russian_cumulative_inflation = 1.0 #Общая инфляция с 1897-го года
russian_social_politics = 0.01 #Показатель "социальности" политики России
russian_med_gdp = 250

#Политические проблемы России
russian_agro_conflict = 0.5 #Разрешённость земельного вопроса
russian_estates = True #Существует ли в России сословная система


class Population:
    def __init__(self, region, nations, row):
        self.pop_by_nations = {}
        self.rel_by_nations = {}
        self.region = region
        self.nations = nations
        self.prev_laboured = -1
        for i in range(0, len(nations)):
            self.pop_by_nations[nations[i]] = []
            current_nation_pop = row[15 + i]
            men = region.men_pop / region.population
            women = 1.0 - men
            if pd.isna(current_nation_pop):
                for k in range(0, 10):
                    self.pop_by_nations[nations[i]].append([0.0, 0.0])
                continue
            age_weight = [15]
            for k in range(1, 120):
                age_weight.append(age_weight[k - 1] * (0.998 - 0.0005 * k - 0.001 * max(0, k - 40)
                                                       - 0.005 * max(0, k - 60)))
            total_weights = sum(age_weight)
            for k in age_weight:
                p = k / total_weights
                cur_pop = current_nation_pop * p
                self.pop_by_nations[nations[i]].append([int(cur_pop * men), int(cur_pop * women)])

    def natural_growth(self):
        medicine = self.region.medicine_quality
        '''Медицина - важнейший параметр для роста населения. Она снижает смертность при родах, младенческую смертность,
        а также снижает "текущую" смертность от несчастных случаев и старения'''
        town_pop = self.region.town_pop
        '''Доля городских жителей определяет влияние эффектов демографического перехода. Поначалу горожане заводят детей
        реже селян, но со временем для обоих групп эффекты перехода будут проявляться в равной мере'''
        dtt = self.region.dem_transition_towns
        dtr = self.region.dem_transition_rural
        real_dt = town_pop * dtt + (1 - town_pop) * dtr
        '''Выгрузка текущей величины эффекта демографического перехода для горожан и селян'''
        region_stability = self.region.stability
        region_iswar = self.region.iswar
        region_hunger = self.region.hunger
        '''Низкая стабильность и нахождение в состоянии войны снижают желание людей заводить детей.
        Голод помимо понижения желания заводить детей также быстро сокращает население'''
        region_isrussian = self.region.isrussian
        if region_isrussian == 1:
            region_stability *= (russian_stability**0.5)
        '''Должны ли применяться к региону модификаторы нахождения в составе России'''
        region_prosperity = self.region.person_prosperity
        '''Достаток семей. Желание иметь детей наивысшее у людей с низким уровнем дохода, однако этот параметр имеет
        самую низкую важность'''

        #Посчитать доли национальностей
        nations_perc_pop = {}
        for i in self.nations:
            nations_perc_pop[i] = 0
            for j in self.pop_by_nations[i]:
                nations_perc_pop[i] += j[0] + j[1]
            if nations_perc_pop[i] == 0.0:
                nations_perc_pop.pop(i)
                continue
            nations_perc_pop[i] /= self.region.population
        perc_pop_keys = list(nations_perc_pop.keys())
        perc_pop_keys.sort(key=lambda x: -nations_perc_pop[x])

        #Шаг 1. "Состарить" население на месяц
        for i in self.nations:
            lnn = len(self.pop_by_nations[i])
            k = lnn
            while k > 0:
                k -= 1
                men = self.pop_by_nations[i][k][0]
                women = self.pop_by_nations[i][k][1]
                if men + women == 0.0:
                    continue
                #Мужчины
                elder_risk = max(0, k - 45 - medicine**0.7) * 0.0001 #Шанс умереть от болезней, связанных со старением
                #Шанс умереть от несчастного случая/болезни наивысший в подростковом возрасте и в молодости
                #Также этот эффект отражает шанс умереть младенцем
                basic_risk = 0.0
                if k == 0:
                    basic_risk = 0.015 - max(0.0125, 0.0005 * medicine)
                if 12 >= k > 0:
                    basic_risk = 0.005 - max(0.00491, 0.0001 * medicine)
                if 28 > k > 12:
                    basic_risk = 0.0009 - max(0.00076, 0.00002 * medicine)
                if k >= 28:
                    basic_risk = 0.0001
                #Эффекты стабильности
                stability_risk = 0.00001 + 0.001 * (1 - region_stability)
                #Эффекты войны
                war_risk = max(0.0005 * region_iswar, 0.012 * region_iswar * int(bool(45 >= k >= 18)))
                #Эффекты голода
                hunger_risk = 0.00008 * region_hunger * self.region.hungry_poverty
                if k < 2:
                    hunger_risk *= 5.5
                if 2 <= k < 7:
                    hunger_risk *= 4.0
                if 7 <= k < 12:
                    hunger_risk *= 2.0
                if 12 <= k < 27:
                    hunger_risk *= 0.6
                if 27 <= k < 45:
                    hunger_risk *= 0.75
                if 45 <= k:
                    hunger_risk *= 1.5
                men_dec = men / 12
                men_inc = (men / 12) * (1 - elder_risk - basic_risk - stability_risk - war_risk - hunger_risk)
                if men_dec < 1 and men > 0:
                    men_dec = np.random.choice([0, 1], p=(1 - men_dec, men_dec))
                if int(men_dec) == int(men_inc):
                    men_inc -= np.random.choice([0, 1],
                                                p=(1 - elder_risk - basic_risk - stability_risk -
                                                   war_risk - hunger_risk,
                                                   elder_risk + basic_risk + stability_risk +
                                                   war_risk + hunger_risk))
                men_dec = int(men_dec)
                men_inc = int(men_inc)
                self.pop_by_nations[i][k][0] -= min(self.pop_by_nations[i][k][0], men_dec)
                if k == lnn - 1:
                    self.pop_by_nations[i].append([men_inc, 0])
                    lnn += 1
                else:
                    self.pop_by_nations[i][k + 1][0] += men_inc

                #Женщины (эффекты от старения понижены, эффекты несчастных случаев понижены, эффекты войны понижены)
                elder_risk = max(0, k - 50 - medicine**0.7) * 0.00008 #Шанс умереть от болезней, связанных со старением
                # Шанс умереть от несчастного случая наивысший в подростковом возрасте и в молодости
                # Также этот эффект отражает шанс умереть младенцем
                basic_risk = 0.0
                if k == 0:
                    basic_risk = 0.02 - max(0.0175, 0.0005 * medicine)
                if 12 >= k > 0:
                    basic_risk = 0.004 - max(0.00391, 0.0001 * medicine)
                if 28 > k > 12:
                    basic_risk = 0.001 - max(0.0009, 0.00005 * medicine)
                if k >= 28:
                    basic_risk = 0.0001
                # Эффекты стабильности
                stability_risk = 0.00001 + 0.001 * (1 - region_stability)
                # Эффекты войны
                war_risk = 0.0005
                # Эффекты голода
                hunger_risk = 0.00008 * region_hunger * self.region.hungry_poverty
                if k < 2:
                    hunger_risk *= 5.5
                if 2 <= k < 7:
                    hunger_risk *= 4.0
                if 7 <= k < 12:
                    hunger_risk *= 2.0
                if 12 <= k < 27:
                    hunger_risk *= 0.6
                if 27 <= k < 45:
                    hunger_risk *= 0.75
                if 45 <= k:
                    hunger_risk *= 1.5
                women_dec = int(women / 12)
                women_inc = int((women / 12) * (1 - elder_risk - basic_risk - stability_risk - war_risk - hunger_risk))
                if women_dec < 1 and women > 0:
                    women_dec = np.random.choice([0, 1], p=(1 - women_dec, women_dec))
                if int(women_dec) == int(women_inc):
                    women_inc -= np.random.choice([0, 1], p=(1 - elder_risk - basic_risk - stability_risk -
                                                             war_risk - hunger_risk,
                                                             elder_risk + basic_risk + stability_risk +
                                                             war_risk + hunger_risk))
                women_dec = int(women_dec)
                women_inc = int(women_inc)
                self.pop_by_nations[i][k][1] -= min(self.pop_by_nations[i][k][1], women_dec)
                if k == lnn - 1:
                    self.pop_by_nations[i].append([0, women_inc])
                    lnn += 1
                else:
                    self.pop_by_nations[i][k + 1][1] += women_inc

        #Шаг 2. "Родить" новых граждан, "умертвить" при родах часть женщин
        #Первичный расчёт граждан репродуктивного возраста
        rep_men = 0 #Количество мужчин репродуктивного возраста
        rep_women = 0 #Количество женщин репродуктивного возраста
        for i in self.nations:
            k = min(51, len(self.pop_by_nations[i]))
            while k > 14:
                k -= 1
                kf = 1
                if k < 16:
                    kf = 0.25
                if 16 <= k < 18:
                    kf = 0.5
                if 35 <= k < 40:
                    kf = 0.7
                if 40 <= k:
                    kf = 0.25
                rep_men += self.pop_by_nations[i][k][0] * kf
                rep_women += self.pop_by_nations[i][k][1] * kf
        k_chance = max(1.0, (rep_men / rep_women) ** 0.5) #Коэффициент "покрытия" женщин мужчинами
        for i in self.nations:
            k = min(51, len(self.pop_by_nations[i]))
            while k > 14:
                k -= 1
                basic_chance = 0.0 #Базовый шанс родить ребёнка в этом месяце
                preg_risk = 0.0 #Риск умереть при родах
                if k < 16:
                    basic_chance = 0.00067
                    preg_risk = 0.2 - min(0.19, 0.004 * medicine)
                if 16 <= k < 21:
                    basic_chance = 0.0067
                    preg_risk = 0.1 - min(0.095, 0.008 * medicine)
                if 21 <= k < 35:
                    basic_chance = 0.0067
                    preg_risk = 0.05 - min(0.048, 0.008 * medicine)
                if 35 <= k < 40:
                    basic_chance = 0.003
                    preg_risk = 0.1 - min(0.095, 0.006 * medicine)
                if 40 <= k:
                    basic_chance = 0.001
                    preg_risk = 0.2 - min(0.19, 0.006 * medicine)
                if region_iswar == 1:
                    basic_chance *= 0.5
                if region_hunger == 1:
                    basic_chance *= 0.3
                if self.region.med_overpopulation > 1.0:
                    basic_chance /= self.region.med_overpopulation
                if self.region.agro_overpopulation > 1.0:
                    basic_chance /= self.region.agro_overpopulation
                basic_chance *= k_chance
                basic_chance *= (region_stability + 0.3)**0.8
                basic_chance *= 3 - (2.8 * (real_dt**0.3))
                if region_prosperity < -100.0:
                    basic_chance *= 2 - (1 - real_dt**0.3)
                if -100.0 <= region_prosperity < -50.0:
                    basic_chance *= 1.5 - (0.5 - real_dt**0.3 * 0.5)
                if -50.0 <= region_prosperity < 0:
                    basic_chance *= 1.2 - (0.2 - real_dt**0.3 * 0.2)
                birth_count = self.pop_by_nations[i][k][1] * basic_chance
                if birth_count < 0:
                    birth_count = 0
                if 0.0 < birth_count < 1.0:
                    birth_count = np.random.choice([0, 1], p=(1 - birth_count, birth_count))
                    if birth_count == 1:
                        self.pop_by_nations[i][k][1] -= np.random.choice([0, 1], p=(1 - preg_risk, preg_risk))
                if birth_count >= 2.0:
                    self.pop_by_nations[i][0][0] += int(birth_count / 2)
                    self.pop_by_nations[i][0][1] += int(birth_count - int(birth_count / 2))
                    self.pop_by_nations[i][k][1] -= int(birth_count * preg_risk)
                if birth_count == 1.0:
                    if np.random.choice([0, 1], p=(0.5, 0.5)) == 0:
                        self.pop_by_nations[i][0][0] += 1
                    else:
                        self.pop_by_nations[i][0][1] += 1
                    self.pop_by_nations[i][k][1] -= np.random.choice([0, 1], p=(1 - preg_risk, preg_risk))

        #Шаг 3. Ассимилировать часть граждан в другие народы
        for i in self.nations:
            assim_chance = 0
            for j in perc_pop_keys:
                assim_chance = nations_perc_pop[j] * assim_data.at[i, j] * 0.005
                if j == self.region.state_nation:
                    assim_chance *= (1 + self.region.assimilation)
                if assim_chance == 0:
                    continue
                k = min(41, len(self.pop_by_nations[i]))
                while k > 12:
                    k -= 1
                    age_k = 0.0
                    if 12 <= k < 16:
                        age_k = 1.25
                    else:
                        if 16 <= k < 21:
                            age_k = 1.0
                        else:
                            if 21 <= k < 28:
                                age_k = 0.7
                            else:
                                if 28 <= k < 35:
                                    age_k = 0.35
                                else:
                                    if 35 <= k:
                                        age_k = 0.1
                    for sex in [0, 1]:
                        target_inc = self.pop_by_nations[i][k][sex] * assim_chance * age_k
                        if target_inc == 0:
                            continue
                        if 0 < target_inc < 1:
                            target_inc = np.random.choice([0, 1], p=(1 - target_inc, target_inc))
                            self.pop_by_nations[i][k][sex] -= target_inc
                            self.pop_by_nations[j][k][sex] += target_inc
                        if target_inc >= 1:
                            target_inc = int(target_inc)
                            self.pop_by_nations[i][k][sex] -= target_inc
                            self.pop_by_nations[j][k][sex] += target_inc

        #Шаг 4. Пересчитать население
        men_pop = sum([sum([self.pop_by_nations[k1][k2][0] for k2 in range(0, len(self.pop_by_nations[k1]))])
                          for k1 in self.pop_by_nations.keys()])
        women_pop = sum([sum([self.pop_by_nations[k1][k2][1] for k2 in range(0, len(self.pop_by_nations[k1]))])
                          for k1 in self.pop_by_nations.keys()])
        population = men_pop + women_pop
        self.region.population = population
        self.region.men_pop = men_pop
        self.region.women_pop = women_pop

    def return_labour(self):
        gdp = 0.0
        gdp_k = 1.0
        laboured = 0
        for i in self.nations:
            k = min(russian_unlabour + 1, len(self.pop_by_nations[i]))
            while k > russian_child:
                k -= 1
                if k < 21:
                    gdp += self.region.gdp_per_person * (self.pop_by_nations[i][k][0] +
                                                         self.pop_by_nations[i][k][1]) * 0.7
                    laboured += (self.pop_by_nations[i][k][0] + self.pop_by_nations[i][k][1]) * 0.7
                if 21 <= k < 50 + int(self.region.medicine_quality**0.7):
                    gdp += self.region.gdp_per_person * (self.pop_by_nations[i][k][0] +
                                                         self.pop_by_nations[i][k][1])
                    laboured += (self.pop_by_nations[i][k][0] + self.pop_by_nations[i][k][1])
                if 50 + int(self.region.medicine_quality**0.7) <= k:
                    gdp += self.region.gdp_per_person * (self.pop_by_nations[i][k][0] +
                                                         self.pop_by_nations[i][k][1]) * 0.5
                    laboured += (self.pop_by_nations[i][k][0] + self.pop_by_nations[i][k][1]) * 0.5
            if self.prev_laboured == -1:
                gdp_k = 1.0
            else:
                if laboured > self.prev_laboured:
                    gdp_k = ((self.prev_laboured / laboured) + 1) / 2
                else:
                    if laboured != 0:
                        gdp_k = ((self.prev_laboured / laboured) + 0.5) / 1.5
                    else:
                        gdp_k = 1.0
            self.prev_laboured = laboured
            self.region.unlaboured = self.region.population - laboured
        return int(gdp * gdp_k)


class State:
    def __init__(self):
        #Параметры государства
        self.toleracy = 0.2  #Терпимость к меньшинствам
        self.openness = 1.0  #Открытость границ
        self.stability = 0.8  #Стабильность в стране
        self.democracy = 0.0  #Демократичность режима
        self.unlabour = 90  # Возраст выхода на пенсию
        self.child = 12  # Возраст, с которого разрешён труд
        self.basic_cost = 5.0  # Базовая стоимость необходимых для жизни продуктов
        self.monthly_inflation = 1.001  # Месячная инфляция
        self.social_politics = 0.01  # Показатель "социальности" политики
        self.med_gdp = 250

        #Список регионов
        self.regions = {}

class Region:
    def __init__(self, nations, row):
        #Неизменяемые параметры:
        self.name = row[0]
        self.img_color = row[len(row) - 1]
        self.square = img_blue_colors[self.img_color]

        #Природные параметры:
        self.climate = row[10]

        #Экономические параметры:
        self.gdp_per_person = row[1]
        self.region_gdp = row[2]
        self.person_prosperity = row[3]
        self.life_cost = row[4]
        self.stratification = row[5] * 0.01
        self.infrastructure = row[6]
        self.industry = 0.0 #Уровень индустриализации
        self.postindustry = 0.0 #Уровень сферы услуг

        #Демографические параметры:
        self.population = row[11]
        self.men_pop = row[13]
        self.women_pop = row[14]
        self.town_pop = row[12]
        self.literacy = row[7] * 0.01
        self.education_quality = row[8]
        self.medicine_quality = row[9]
        self.migration_attract = 1.0 #Привлекательность региона для миграции
        self.pop_density = self.population / self.square
        self.dem_transition_towns = 0.3 #Выраженность демографического перехода в городах
        self.dem_transition_rural = 0.0 #Выраженность демографического перехода в деревне
        self.population_object = Population(self, nations, row)
        self.hunger = 0 #Есть ли в регионе ГОЛОД
        self.product_dotation = 0 #Дотации на продукты для борьбы с голодом

        #Параметры перенаселения
        self.fact_town_pop = int(self.population * self.town_pop) #Численность городского населения
        self.fact_agro_pop = self.population - self.fact_town_pop #Численность сельского населения
        self.town_effective = self.square * global_building_technology * 1000
        self.agro_effective = self.square * (self.climate ** 2) * 50 / global_agro_technology
        self.agro_overpopulation = max(1.0, self.fact_agro_pop / self.agro_effective)
        self.town_overpopulation = max(1.0, self.fact_town_pop / self.town_effective)
        self.med_overpopulation = (self.agro_overpopulation * self.fact_agro_pop +
                                   self.town_overpopulation * self.fact_town_pop) / self.population

        #Политические параметры:
        self.isrussian = 1
        self.state = None
        self.iswar = 0 #Находится ли регион в составе государства, ведущего войну
        self.openness = 0.6
        self.governor_eff = 1
        self.stability = 1
        self.state_nation = 'Русские'
        self.assimilation = 0.4
        self.industry_efforts = 0.0 #Усилия по индустриализации
        self.monarch_power = 0.5 #Поддержка монархии
        self.autocrat_power = 0.5 #Поддержка авторитаризма
        self.democracy_power = 0.3 #Поддержка демократии
        self.natpop_power = 0.0  #Поддержка национал-популизма
        self.liberal_power = 0.4 #Поддержка либерализма
        self.socialist_power = 0.5 #Поддержка социализма
        self.communist_power = 0.1 #Поддержка коммунизма
        self.conserv_power = 0.7 #Поддержка традиционных ценностей
        self.progress_power = 0.2 #Поддержка прогрессивных ценностей

        # Параметры уровня жизни
        self.housing = int(0.9 * self.population)
        self.arenda = self.gdp_per_person * (self.population / self.housing) * 0.1
        self.product_cost = ((russian_basic_cost / self.infrastructure) *
                             (1 + (self.gdp_per_person / russian_basic_cost**2)))
        self.poverty = 0
        self.hungry_poverty = 0
        self.population_object.return_labour()
        self.unlaboured = 0 #Численность иждивенцев

    def overpop(self):
        #Проверка на перенаселение
        self.fact_town_pop = int(self.population * self.town_pop)  # Численность городского населения
        self.fact_agro_pop = self.population - self.fact_town_pop  # Численность сельского населения
        self.town_effective = self.square * global_building_technology * 1000
        self.agro_effective = self.square * (self.climate ** 2) * 50 / global_agro_technology * russian_agro_conflict
        self.agro_overpopulation = max(1.0, self.fact_agro_pop / self.agro_effective)
        self.town_overpopulation = max(1.0, self.fact_town_pop / self.town_effective)
        self.med_overpopulation = (self.agro_overpopulation * self.fact_agro_pop +
                                   self.town_overpopulation * self.fact_town_pop) / self.population
        if self.agro_overpopulation > 1.0:
            self.stability = max(0.0, self.stability - (self.agro_overpopulation - 1) * 0.02)
        if self.town_overpopulation > 1.0:
            self.stability = max(0.0, self.stability - (self.agro_overpopulation - 1) * 0.01)

    def natural_growth(self):
        #Снижение стабильности в случае голода
        self.overpop()
        if self.stability < 1.0:
            self.stability = min(1.0, self.stability + 0.02)
        if self.hunger == 1:
            if self.stability > 0.25:
                self.stability -= 0.05
        self.gdp_per_person = max(1.0, self.gdp_per_person)
        self.population_object.natural_growth()
        self.pop_density = self.population / self.square

    def economy_growth(self):
        #Эффективность труда (изменяется в пределах от -3% до +3% в месяц
        mod = 0.0
        #Базовое развитие (зависит от наличия инфляции) (мин - -0.5, макс - 0.1)
        if self.isrussian == 1:
            inflation = russian_monthly_inflation
        else:
            inflation = self.state.monthly_inflation
        if 1.005 > inflation >= 1.0:
            mod += 0.1
        else:
            if inflation < 1.0:
                mod -= 0.1
            else:
                if inflation >= 1.005:
                    mod -= 0.1
                    if inflation > 1.01:
                        mod -= 0.1
                        if inflation > 1.03:
                            mod -= 0.1
                            if inflation > 1.1:
                                mod -= 0.1
                                if inflation > 1.2:
                                    mod -= 0.1

        #Развитие благодаря открытости и инвестициям (мин - 0, макс - 0.3)
        if self.isrussian == 1:
            openness = russian_openness
        else:
            openness = self.state.openness
        if self.gdp_per_person < global_gdp_med * 0.8 and inflation < 1.01:
            mod += 0.1 * openness / inflation
            if self.gdp_per_person < global_gdp_med * 0.5 and inflation < 1.008:
                mod += 0.1 * openness / inflation
                if self.gdp_per_person < global_gdp_med * 0.2 and inflation < 1.006:
                    mod += 0.1 * openness / inflation


        #Развитие благодаря урбанизации (мин - 0, макс - 0.3)
        self.gdp_per_person *= 1 + 0.0005 * self.town_pop
        if self.town_pop > 0.3:
            mod += 0.1
            if self.town_pop > 0.5:
                mod += 0.1
                if self.town_pop > 0.7:
                    mod += 0.1

        #Развитие благодаря грамотности (мин - -0.3, макс - 0.4)
        if self.literacy < 0.2:
            mod -= 0.1
            if self.literacy < 0.1:
                mod -= 0.1
                if self.literacy < 0.05:
                    mod -= 0.1
        else:
            if self.literacy > 0.5:
                mod += 0.1
                if self.literacy > 0.9:
                    mod += 0.1
            if self.education_quality > 5.0:
                mod += 0.1
                if self.education_quality > 9.0:
                    mod += 0.1

        #Развитие благодаря переходным эффектам экономики
        i_value = (self.industry_efforts * self.town_pop) * calculate_basic_value(self.industry)
        self.industry += i_value

        if self.industry < 0.2:
            mod += 30 * i_value
        else:
            if self.industry < 0.4:
                mod += 40 * i_value
            else:
                if self.industry < 0.6:
                    mod += 50 * i_value
                else:
                    if self.industry < 0.95:
                        mod += 60 * i_value

        if year >= 1960:
            pi_value = (self.industry * self.town_pop * min(0.0, max(1.0, self.person_prosperity / 500))) * \
                       calculate_basic_value(self.postindustry)
            self.postindustry += pi_value

            if self.postindustry < 0.3:
                mod += 10 * pi_value
            else:
                if self.postindustry < 0.6:
                    mod += 15 * pi_value
                else:
                    if self.postindustry < 0.95:
                        mod += 20 * pi_value

        #Развитие благодаря эффективности губернатора (мин - -0.1, макс - 0.2)
        if self.governor_eff < 0.6:
            mod -= 0.1
        else:
            if self.governor_eff >= 1.0:
                mod += 0.1
                if self.governor_eff >= 1.5:
                    mod += 0.1

        #Развитие благодаря инфраструктуре
        if self.infrastructure < 0.3:
            mod -= 0.1
        else:
            if self.infrastructure >1.0:
                mod += 0.1
                if self.infrastructure > 3.0:
                    mod += 0.1
                    if self.infrastructure > 5.0:
                        mod += 0.1
                        if self.infrastructure > 7.0:
                            mod += 0.1
        if self.infrastructure > (1.0 + int(10 * global_building_technology ** 0.5) / 10) * 0.7:
            mod += 0.1

        #Итоговый подсчёт
        if mod > 0:
            value = 1 + 3 * calculate_basic_value(1 - max(1.0, mod / 5))
        else:
            value = 1 - 3 * calculate_basic_value(1 - max(1.0, abs(mod) / 5))

        self.gdp_per_person *= value

        #Пересчёт стоимости жизни и благосостояния
        self.gdp_per_person = max(1.0, self.gdp_per_person)
        self.arenda = 10 * (russian_basic_cost * (self.population / self.housing) * self.infrastructure *
                       (global_gdp_med / self.gdp_per_person) * (self.medicine_quality / global_max_med) *
                       (self.education_quality / 5) ** 0.5 * (1 - self.stratification))
        self.product_cost = self.med_overpopulation * russian_basic_cost / self.infrastructure
        gdp_life_k = min(1.0, self.gdp_per_person / global_gdp_med)
        self.product_cost *= gdp_life_k
        self.product_cost = max(0.0, self.product_cost - self.product_dotation)
        if self.pop_density < 5:
            cost_mod = 1 - self.pop_density / 5
            self.product_cost *= cost_mod
        self.life_cost = self.arenda + self.product_cost + self.product_cost * (self.unlaboured /
                                                                                (self.population - self.unlaboured))
        self.life_cost *= russian_monthly_inflation ** 0.5
        s = min(1.0, max(0.0, self.stratification))
        if s != 0.0:
            prosperity_median = (s - 1 - (1 + s**2)**0.5) / 2 * s
        else:
            prosperity_median = 0.5
        self.person_prosperity = (0.5 * self.gdp_per_person * prosperity_median - self.life_cost)
        g = self.gdp_per_person
        l = self.life_cost * 4
        self.poverty = 100 * (l / g)
        if self.poverty > 99.9:
            self.poverty = 99.9
        if self.poverty < 0:
            self.poverty = 0
        l = (self.life_cost - self.arenda) * 0.25
        self.hungry_poverty = 100 * (l / g)
        if self.hungry_poverty > 99.9:
            self.hungry_poverty = 99.9
        if self.hungry_poverty < 0:
            self.hungry_poverty = 0
        if self.hungry_poverty > 10.0:
            self.hunger = 1
        if self.hungry_poverty < 7.0:
            self.hunger = 0
        #Стратификация и социальная политика
        value_plus = calculate_basic_value(self.stratification)
        value_minus = calculate_basic_value(1 - self.stratification)
        mod = 0.0

        if self.poverty < 10:
            mod += 0.1
        if self.poverty > 70:
            mod += 0.1
            if self.poverty > 90:
                mod += 0.1
        if self.hungry_poverty > 5:
            mod += 0.1
            if self.hungry_poverty > 10:
                if self.gdp_per_person > 5.0:
                    self.gdp_per_person -= 0.5
                    self.product_dotation += 1.0
            if self.hungry_poverty > 30:
                mod += 0.1
        else:
            if self.product_dotation > 0.0:
                self.gdp_per_person += min(0.25, self.product_dotation) * 0.2
                self.product_dotation = max(0.0, self.product_dotation - 0.25)
        if self.product_dotation > 0.0:
            mod -= 0.1
            if self.product_dotation > 3.0:
                mod -= 0.1
                if self.product_dotation > 5.0:
                    mod -= 0.1
        if russian_monthly_inflation > 1.005:
            mod += 0.1
        if russian_monthly_inflation < 1.0:
            mod += 0.1
        if self.isrussian == 1:
            soc_pol = russian_social_politics
        else:
            soc_pol = self.state.social_politics
        mod -= soc_pol * 5

        if mod > 0:
            self.stratification += mod * value_plus
        if mod < 0:
            self.stratification -= mod * value_minus
        #Итоговый подсчёт
        self.gdp_per_person = max(1.0, self.gdp_per_person)
        self.region_gdp = self.population_object.return_labour()

    def literacy_and_medicine(self):
        #Грамотность
        value_plus = calculate_basic_value(self.literacy)
        value_minus = calculate_basic_value(1 - self.literacy)
        mod = 0.0
        #Рост грамотности от урбанизации
        if self.town_pop > 0.4:
            mod += 0.1
            if self.town_pop > 0.9:
                mod += 0.1
        if self.education_quality > 1.0:
            mod += 0.1
            if self.education_quality > 3.0:
                mod += 0.1
                if self.education_quality > 5.0:
                    mod += 0.1
        if self.gdp_per_person > 200:
            mod += 0.1
            if self.gdp_per_person > 500:
                mod += 0.1
        if self.poverty < 30:
            mod += 0.1
            if self.poverty < 10:
                mod += 0.1
        if self.literacy < 0.1:
            mod += 0.1

        if self.town_pop < 0.1:
            mod -= 0.1
        if self.gdp_per_person < 50:
            mod -= 0.1
        if self.hunger == 1 or self.iswar == 1:
            mod -= 0.1
        if self.literacy > 0.95:
            mod -= 0.1

        if mod > 0:
            self.literacy += value_plus * mod
        if mod < 0:
            self.literacy += value_minus * mod

        #Медицина
        value_plus = global_max_med * calculate_basic_value(self.medicine_quality / global_max_med)
        value_minus = global_max_med * calculate_basic_value(1 - (self.medicine_quality / global_max_med))
        mod = 0.0

        if self.literacy > 0.9:
            mod += 0.1
        if self.education_quality > 1.0:
            mod += 0.1
            if self.education_quality > 2.5:
                mod += 0.1
                if self.education_quality > 5.0:
                    mod += 0.1
                    if self.education_quality > 8.0:
                        mod += 0.1
                        if self.education_quality > 9.5:
                            mod += 0.1
        if self.gdp_per_person > global_gdp_med:
            mod += 0.1
            if self.gdp_per_person > global_gdp_med * 2:
                mod += 0.1
        if self.town_pop > 0.6:
            mod += 0.1
        if self.person_prosperity > 300:
            mod += 0.1

        if self.literacy < 0.1:
            mod -= 0.1
        if self.education_quality < 0.5:
            mod -= 0.1
        if self.person_prosperity < 0:
            mod -= 0.1

        if mod > 0:
            self.medicine_quality += value_plus * mod
        if mod < 0:
            self.medicine_quality += value_minus * mod

        #Качество образования
        value_plus = 10 * calculate_basic_value(self.education_quality / 10)
        value_minus = 10 * calculate_basic_value(1 - (self.education_quality / 10))
        mod = -0.2

        if self.gdp_per_person > global_gdp_med * 0.5:
            mod += 0.1
            if self.gdp_per_person > global_gdp_med * 0.8:
                mod += 0.1
                if self.gdp_per_person > global_gdp_med:
                    mod += 0.1
                    if self.gdp_per_person > global_gdp_med * 1.4:
                        mod += 0.1
                        if self.gdp_per_person > global_gdp_med * 1.7:
                            mod += 0.1
        if self.town_pop > 0.4:
            mod += 0.1
            if self.town_pop > 0.7:
                mod += 0.1
        if self.literacy > 0.9:
            mod += 0.1
        if self.person_prosperity > 300:
            mod += 0.1
            if self.person_prosperity > 500:
                mod += 0.1

        if self.gdp_per_person < global_gdp_med * 0.1:
            mod -= 0.1
        if self.literacy < 0.1:
            mod -= 0.1
        if self.person_prosperity < 0:
            mod -= 0.1

        if mod > 0:
            self.education_quality += value_plus * mod
        if mod < 0:
            self.education_quality += value_minus * mod

    def urban(self):
        # Рост урбанизации
        s_value_plus = calculate_basic_value(self.town_pop)
        s_value_minus = calculate_basic_value(1 - self.town_pop)
        mod = 0.0

        if self.gdp_per_person > 100:
            mod += 0.1
            if self.gdp_per_person > 500:
                mod += 0.1
            if self.literacy > 0.3:
                mod += 0.1
        if self.town_pop < 0.4:
            if self.literacy > 0.5:
                mod += 0.1
                if self.literacy > 0.9:
                    mod += 0.1
        else:
            if self.literacy > 0.8:
                mod += 0.1
        if global_agro_technology > 2.0:
            mod += 0.1
            if self.agro_overpopulation > 1.2:
                mod += 0.1
        if self.agro_overpopulation > 1.0:
            mod += 0.1
            if self.agro_overpopulation > 1.4:
                mod += 0.1
                if self.agro_overpopulation > 1.8:
                    mod += 0.1

        if self.hunger == 1:
            mod -= 0.1
        if self.climate > 1.5:
            mod -= 0.1
        if self.housing < self.population * 0.8:
            mod -= 0.1
        if self.poverty > 0.4:
            mod -= 0.1
            if self.poverty > 0.8:
                mod -= 0.1

        if mod > 0:
            self.town_pop += s_value_plus * mod
        if mod < 0:
            self.town_pop += s_value_minus * mod

        #Демографический переход городской
        dtt_plus_value = calculate_basic_value(self.dem_transition_towns)
        dtt_minus_value = calculate_basic_value(1 - self.dem_transition_towns)
        mod = 0
        if self.gdp_per_person > 150:
            mod += 0.1
            if self.gdp_per_person > 500:
                mod += 0.1
                if self.gdp_per_person > 2000:
                    mod += 0.1
            if self.literacy > 0.8:
                mod += 0.1
        if self.town_pop > 0.5:
            mod += 0.1
            if self.town_pop > 0.8:
                mod += 0.1
        if self.medicine_quality > 5.0:
            mod += 0.1
            if self.medicine_quality > 12.0:
                mod += 0.1
        if self.person_prosperity > 100:
            mod += 0.1
            if self.person_prosperity > 300:
                mod += 0.1

        if self.hunger == 1:
            mod -= 0.1
        if self.poverty > 0.5:
            mod -= 0.1
            if self.poverty > 0.8:
                mod -= 0.1
        if self.literacy < 0.2:
            mod -= 0.1

        if mod > 0:
            self.dem_transition_towns += dtt_plus_value * mod
        if mod < 0:
            self.dem_transition_towns += dtt_minus_value * mod

        #Демографический переход в деревне
        dtr_plus_value = self.dem_transition_towns * calculate_basic_value(self.dem_transition_rural)
        dtr_minus_value = self.dem_transition_towns * calculate_basic_value(1 - self.dem_transition_rural)
        mod = 0
        if self.gdp_per_person > 200:
            mod += 0.1
            if self.gdp_per_person > 1000:
                mod += 0.1
            if self.town_pop > 0.4:
                mod += 0.1
                if self.town_pop > 0.6:
                    mod += 0.1
        if self.medicine_quality > 7.0:
            mod += 0.1
            if self.medicine_quality > 15.0:
                mod += 0.1
        if self.person_prosperity > 50:
            mod += 0.1
            if self.person_prosperity > 120:
                mod += 0.1
                if self.education_quality > 3.0:
                    mod += 0.1
        if self.agro_overpopulation > 1.0:
            mod += 0.1

        if self.hunger == 1:
            mod -= 0.1
        if self.poverty > 0.6:
            mod -= 0.1
        if self.literacy < 0.2:
            mod -= 0.1

        if mod > 0:
            self.dem_transition_rural += dtr_plus_value * mod
        if mod < 0:
            self.dem_transition_rural += dtr_minus_value * mod


    def building(self):
        #Изменение инфраструктуры
        max_infra = 1.0 + int(10 * global_building_technology ** 0.5) / 10
        value_plus = max_infra * calculate_basic_value(self.infrastructure / max_infra)
        value_minus = max_infra * calculate_basic_value(1 - (self.infrastructure / max_infra))
        mod = 0.0

        if month == 3:
            mod -= 0.1 + 0.1 / self.climate

        spent = 0.0
        infra_build_eff = (global_building_technology ** 0.5) / (self.square ** 0.3)
        for cnt in range(1, 11):
            if self.infrastructure < max_infra * 0.1 * cnt:
                if self.region_gdp * self.governor_eff > 1000000 * cnt * (1 + cnt) / 2:
                    mod += 0.1 * infra_build_eff
                    spent += cnt * 10 / self.governor_eff

        if mod > 0:
            self.infrastructure += mod * value_plus
        if mod < 0:
            self.infrastructure += mod * value_minus

        labour = self.region_gdp / self.gdp_per_person
        self.gdp_per_person = (self.region_gdp - spent) / labour
        #Строительство жилья
        if month == 3:
            if self.housing > 0:
                self.housing = max(0, int(self.housing * (0.9998 - (3 - self.climate) * 0.0001)))
        max_housing = (self.town_effective * self.town_pop) + (self.agro_effective * (1 - self.town_pop))
        if self.housing < self.population and self.housing < max_housing:
            if self.region_gdp <= 10000000:
                self.housing = min(max_housing, self.housing + int(self.population * 0.0002))
            else:
                if self.region_gdp <= 50000000:
                    self.housing = min(max_housing, self.housing + int(self.population * 0.0005))
                    self.gdp_per_person = (self.region_gdp - 1250) / labour
                else:
                    if self.region_gdp <= 250000000:
                        self.housing = min(max_housing, self.housing + int(self.population * 0.001))
                        self.gdp_per_person = (self.region_gdp - 2500) / labour
                    else:
                        self.housing = min(max_housing, self.housing + int(self.population * 0.002))
                        self.gdp_per_person = (self.region_gdp - 5000) / labour

    def political_effects(self):
        if self.isrussian:
            if russian_estates:
                if self.stability > 0.5:
                    self.stability -= 0.01
            if russian_stability < self.stability:
                self.stability -= 0.01
            if russian_democracy < 0.3:
                if np.random.choice([0, 1], p=(0.99995, 0.00005)) == 1:
                    if russian_democracy <= 0.1:
                        self.governor_eff = np.random.choice(
                            a=[0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5],
                            p=(0.01, 0.04, 0.10, 0.25, 0.40, 0.16, 0.04)
                        )
                    if 0.1 < russian_democracy <= 0.3:
                        self.governor_eff = np.random.choice(
                            a=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7],
                            p=(0.04, 0.08, 0.12, 0.40, 0.24, 0.09, 0.03)
                        )
            else:
                event = np.random.choice([0, 1, 2], p=(0.98298, 0.017, 0.00002))
                if event == 1:
                    if self.governor_eff > 1.1:
                        if russian_democracy <= 0.7:
                            if np.random.choice([0, 1], p=(0.6, 0.4)) == 1:
                                self.governor_eff = np.random.choice(
                                    a=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7],
                                    p=(0.03, 0.07, 0.10, 0.40, 0.25, 0.10, 0.05)
                                )
                            else:
                                if self.governor_eff < 2.0:
                                    self.governor_eff = self.governor_eff + 0.1
                        else:
                            if np.random.choice([0, 1], p=(0.8, 0.2)) == 1:
                                self.governor_eff = np.random.choice(
                                    a=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7],
                                    p=(0.03, 0.07, 0.10, 0.40, 0.25, 0.10, 0.05)
                                )
                            if self.governor_eff < 2.0:
                                self.governor_eff = self.governor_eff + 0.1
                if event == 2:
                    self.governor_eff = np.random.choice(
                        a=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7],
                        p=(0.03, 0.07, 0.10, 0.40, 0.25, 0.10, 0.05)
                    )


def calculate_basic_value(x_value):
    #x_value должно принимать значения от 0 до 1
    return min(1.0, max(0.0, 1 - 1.01 ** (x_value ** 0.5 - 1)))


def population_to_str(pop):
    str_pop = ''
    pop = int(pop)
    if pop < 1000:
        str_pop = str(pop)
        return str_pop
    else:
        while pop > 0:
            if str_pop != '':
                to_pop = str(pop % 1000)
                if pop > 1000:
                    while len(to_pop) < 3:
                        to_pop = '0' + to_pop
                str_pop = to_pop + '.' + str_pop
            else:
                to_pop = str(pop % 1000)
                if pop > 1000:
                    while len(to_pop) < 3:
                        to_pop = '0' + to_pop
                str_pop = to_pop
            pop //= 1000
        return str_pop


def date_as_str():
    month_dict = {
        1: 'янв.',
        2: 'фев.',
        3: 'мар.',
        4: 'апр.',
        5: 'май',
        6: 'июн.',
        7: 'июл.',
        8: 'авг.',
        9: 'сен.',
        10: 'окт.',
        11: 'ноя.',
        12: 'дек.'
    }
    return str(year) + ', ' + month_dict[month]


def ethnic_list(regs_dict):
    nations_list = []
    target = len(regs_dict[list(regs_dict.keys())[0]].population_object.pop_by_nations.keys())
    for i in regs_dict.keys():
        for j in regs_dict[i].population_object.pop_by_nations.keys():
            population = 0
            if len(nations_list) < target:
                if len(nations_list) == 0:
                    nations_list.append([j, 0])
                else:
                    if not j in [nl[0] for nl in nations_list]:
                        nations_list.append([j, 0])
            for t in regs_dict[i].population_object.pop_by_nations[j]:
                population += t[0] + t[1]
            for t in range(0, len(nations_list)):
                if nations_list[t][0] == j:
                    nations_list[t][1] += population
    nations_list.sort(key=lambda x: -x[1])
    return nations_list


def region_list(regs_dict):
    regs_list = []
    for i in regs_dict.keys():
        regs_list.append((i, regs_dict[i].population))
    regs_list.sort(key=lambda  x: -x[1])
    return regs_list


def history_sim(regs_dict):
    global russian_monthly_inflation, russian_child, russian_democracy, russian_estates, russian_stability
    global russian_agro_conflict, russian_basic_cost
    #Эпоха монархии
    if 1897 <= year < 1917:
        #До русской революции
        if year < 1905:
            russian_stability += 0.001
            if year == 1900 and month == 1:
                russian_monthly_inflation -= 0.002
            #Русско-японская война
            if year == 1904 and month == 1:
                russian_stability = 0.7
                russian_monthly_inflation += 0.01
        #Русская революция
        if year == 1905:
            russian_stability -= 0.005
            russian_monthly_inflation += 0.002
            if month == 1:
                russian_basic_cost += 0.5
        #Довоенный период
        if 1905 < year < 1914:
            #Столыпинская реформа
            if year < 1911 or (year == 1911 and month <= 9):
                russian_agro_conflict += 0.004
                russian_stability += 0.002
            if year == 1906 and month == 4:
                russian_democracy = 0.2
                russian_basic_cost -= 0.5
                russian_monthly_inflation -= 0.005
                russian_stability += 0.05
            if year == 1910 and month == 1:
                russian_basic_cost -= 0.2
                russian_monthly_inflation -= 0.005
            if year == 1913 and month == 1:
                russian_basic_cost -= 0.1
                russian_monthly_inflation -= 0.005
                russian_stability += 0.01
    #Революции и гражданская война
    if 1917 <= year < 1921:
        return
    #Керенщина
    if 1921 <= year < 1936:
        return
    #Предвоенная эпоха
    if 1936 <= year < 1941:
        return
    #Второй Вельткриг
    if 1941 <= year < 1946:
        return
    #Эпоха Савинкова
    if 1946 <= year < 1959:
        return
    #Петровщина
    if 1959 <= year < 1983:
        return
    #Бодровская оттепель
    if 1983 <= year < 1993:
        return
    #Сытые девяностые
    if 1993 <= year:
        return


def save_image(regs_dict, type='nations'):
    global img_num
    dict_blues = {}
    if type == 'nations':
        directory = "Output/Population/Pop"
        max_pop = -1.0
        max_in_reg = {}
        for i in regs_dict.keys():
            dict_blues[regs_dict[i].img_color] = i
            rus_sum = sum([regs_dict[i].population_object.pop_by_nations['Русские'][k][0] +
                           regs_dict[i].population_object.pop_by_nations['Русские'][k][1]
                           for k in range(0, len(regs_dict[i].population_object.pop_by_nations['Русские']))])
            ukr_sum = sum([regs_dict[i].population_object.pop_by_nations['Украинцы'][k][0] +
                           regs_dict[i].population_object.pop_by_nations['Украинцы'][k][1]
                           for k in range(0, len(regs_dict[i].population_object.pop_by_nations['Украинцы']))])
            bel_sum = sum([regs_dict[i].population_object.pop_by_nations['Белорусы'][k][0] +
                           regs_dict[i].population_object.pop_by_nations['Белорусы'][k][1]
                           for k in range(0, len(regs_dict[i].population_object.pop_by_nations['Белорусы']))])
            max_pop = max(max_pop, rus_sum, ukr_sum, bel_sum)
            max_in_reg[i] = [-1.0, (0, 0, 0)]
            if rus_sum > max_in_reg[i][0]:
                max_in_reg[i][0] = rus_sum
                max_in_reg[i][1] = (200, 30, 120)
            if ukr_sum > max_in_reg[i][0]:
                max_in_reg[i][0] = ukr_sum
                max_in_reg[i][1] = (220, 180, 30)
            if bel_sum > max_in_reg[i][0]:
                max_in_reg[i][0] = bel_sum
                max_in_reg[i][1] = (30, 225, 60)
        img_density = img.copy()
        for x in range(0, img_density.width):
            for y in range(38, img_density.height):
                t = img_density.getpixel((x, y))
                if t[0] == 0 and t[1] == 0:
                    if t[2] in dict_blues.keys():
                        cur_reg = dict_blues[t[2]]
                        red = max_in_reg[cur_reg][1][0]
                        green = max_in_reg[cur_reg][1][1]
                        blue = max_in_reg[cur_reg][1][2]
                        bright = ((max_in_reg[cur_reg][0] ** 0.5) / (max_pop ** 0.5))
                        img_density.putpixel((x, y), (int(red * bright), int(green * bright), int(blue * bright)))
    if type == 'infra':
        directory = "Output/Infra/Inf"
        inf_in_reg = {}
        max_infra = 1.0 + int(10 * global_building_technology ** 0.5) / 10
        for i in regs_dict.keys():
            dict_blues[regs_dict[i].img_color] = i
            inf_in_reg[i] = regs_dict[i].infrastructure
        img_density = img.copy()
        for x in range(0, img_density.width):
            for y in range(38, img_density.height):
                t = img_density.getpixel((x, y))
                if t[0] == 0 and t[1] == 0:
                    if t[2] in dict_blues.keys():
                        cur_reg = dict_blues[t[2]]
                        bright = inf_in_reg[cur_reg] / max_infra
                        if bright < 0:
                            red = 0
                            green = 0
                            blue = 0
                        else:
                            if bright > 1:
                                red = 0
                                green = 0
                                blue = 255
                            else:
                                if bright <= 0.5:
                                    red = 240
                                    green = min(240, int(240 * 2 * bright))
                                    blue = 40
                                else:
                                    red = max(0, int(240 * 2 * (1 - bright)))
                                    green = 240
                                    blue = 40
                        img_density.putpixel((x, y), (int(red), int(green), int(blue)))
    if type == 'poverty':
        directory = "Output/Poverty/Pov"
        hunger = {}
        pov_in_reg = {}
        hung_in_reg = {}
        for i in regs_dict.keys():
            dict_blues[regs_dict[i].img_color] = i
            if regs_dict[i].hunger == 1:
                hunger[i] = True
            else:
                hunger[i] = False
            pov_in_reg[i] = regs_dict[i].poverty
            hung_in_reg[i] = regs_dict[i].hungry_poverty
        img_density = img.copy()
        for x in range(0, img_density.width):
            for y in range(38, img_density.height):
                t = img_density.getpixel((x, y))
                if t[0] == 0 and t[1] == 0:
                    if t[2] in dict_blues.keys():
                        cur_reg = dict_blues[t[2]]
                        if hunger[cur_reg]:
                            bright = 1 - (hung_in_reg[cur_reg] / 100)
                            red = 200 * bright
                            green = 0
                            blue = 0
                        else:
                            if pov_in_reg[cur_reg] >= 20:
                                bright = 1 - (pov_in_reg[cur_reg] - 20) / 80
                                red = 220 - 40 * bright
                                green = 90 + 160 * bright
                                blue = 20
                            else:
                                bright = 1 - pov_in_reg[cur_reg] / 20
                                red = 80 - 80 * bright
                                green = 255
                                blue = 30 + 30 * bright
                        img_density.putpixel((x, y), (int(red), int(green), int(blue)))
    if type == 'gdp':
        directory = "Output/GDP/GDP"
        gdp_in_reg = {}
        max_gdp = 0
        for i in regs_dict.keys():
            dict_blues[regs_dict[i].img_color] = i
            gdp_in_reg[i] = regs_dict[i].gdp_per_person
            if gdp_in_reg[i] > max_gdp:
                max_gdp = gdp_in_reg[i]
        img_density = img.copy()
        for x in range(0, img_density.width):
            for y in range(38, img_density.height):
                t = img_density.getpixel((x, y))
                if t[0] == 0 and t[1] == 0:
                    if t[2] in dict_blues.keys():
                        cur_reg = dict_blues[t[2]]
                        if gdp_in_reg[cur_reg] > global_gdp_med:
                            bright = (gdp_in_reg[cur_reg] / max_gdp) ** 0.5
                            red = 30
                            green = 30 + 225 * bright
                            blue = 30
                        else:
                            if gdp_in_reg[cur_reg] >= global_gdp_med / 2:
                                bright = gdp_in_reg[cur_reg] / global_gdp_med
                                red = 220 - 80 * bright
                                green = 180 + 70 * bright
                                blue = 30
                            else:
                                bright = 2 * gdp_in_reg[cur_reg] / global_gdp_med
                                red = 220 * bright
                                green = 0
                                blue = 0
                        img_density.putpixel((x, y), (int(red), int(green), int(blue)))
    if type == 'urban':
        directory = "Output/Urban/Urb"
        urban_in_reg = {}
        for i in regs_dict.keys():
            dict_blues[regs_dict[i].img_color] = i
            urban_in_reg[i] = regs_dict[i].town_pop
        img_density = img.copy()
        for x in range(0, img_density.width):
            for y in range(38, img_density.height):
                t = img_density.getpixel((x, y))
                if t[0] == 0 and t[1] == 0:
                    if t[2] in dict_blues.keys():
                        cur_reg = dict_blues[t[2]]
                        bright = urban_in_reg[cur_reg]
                        red = 250 * bright
                        green = 250 * bright
                        blue = 250 * bright
                        img_density.putpixel((x, y), (int(red), int(green), int(blue)))
    draw = ImageDraw.Draw(img_density)
    fnt = ImageFont.truetype("calibri.ttf", 35)
    total_pop = sum([regs_dict[i].population for i in regs_dict.keys()])
    draw.text((30, 1), population_to_str(total_pop), font=fnt, fill=(0, 0, 0))
    draw.text((280, 1), population_to_str(sum([regs_dict[i].region_gdp * russian_cumulative_inflation
                                               for i in regs_dict.keys()])), font=fnt, fill=(0, 0, 0))
    draw.text((596, 1), str(int(sum([regs_dict[i].poverty * regs_dict[i].population for i in regs_dict.keys()]) /
                                total_pop))
              + '%', font=fnt, fill=(0, 0, 0))
    draw.text((717, 1), str(int(sum([regs_dict[i].literacy * regs_dict[i].population for i in regs_dict.keys()]) * 100 /
                                total_pop))
              + '%', font=fnt, fill=(0, 0, 0))
    draw.text((840, 1), str(int(sum([regs_dict[i].town_pop * regs_dict[i].population for i in regs_dict.keys()]) * 100 /
                                total_pop))
              + '%', font=fnt, fill=(0, 0, 0))
    nations_list = ethnic_list(regs_dict)
    pos = 79
    gray = 0
    fnt_small = ImageFont.truetype("calibri.ttf", 18)
    for i in nations_list[:12]:
        draw.text((1773, pos), i[0] + ': ' + population_to_str(i[1]), font=fnt_small, fill=(gray, gray, gray))
        pos += 22
        gray += 7
    regs_list = region_list(regs_dict)
    pos = 379
    gray = 0
    for i in range(0, 11):
        draw.text((1773, pos), str(i + 1) + '. ' + regs_list[i][0], font=fnt_small, fill=(gray, gray, gray))
        draw.text((1773, pos + 20), 'Население: ' + population_to_str(regs_list[i][1]), font=fnt_small,
                  fill=(gray + 50, gray + 50, gray + 50))
        pos += 42
        gray += 3
    draw.text((1773, 1), date_as_str(), font=fnt, fill=(0, 0, 0))
    img_density.save(directory + str(img_num[type]) + '.png', 'PNG')
    img_num[type] += 1


def main():
    data = pd.read_excel('RegData.xlsx').drop(['Итого', 'Столица субъекта'], axis=1)
    if not os.path.exists('Output/'):
        os.mkdir('Output')
    if not os.path.exists('Output/Population/'):
        os.mkdir('Output/Population')
    if not os.path.exists('Output/Infra/'):
        os.mkdir('Output/Infra')
    if not os.path.exists('Output/Poverty/'):
        os.mkdir('Output/Poverty')
    if not os.path.exists('Output/GDP/'):
        os.mkdir('Output/GDP')
    if not os.path.exists('Output/Urban/'):
        os.mkdir('Output/Urban')
    nations = data.columns.to_list()
    nations = nations[15:len(nations) - 1]
    regs_dict = {}
    for i in data.iterrows():
        row = i[1].to_list()
        regs_dict[row[0]] = Region(nations, row)
    global month, year, global_gdp_med, global_max_med, russian_cumulative_inflation, russian_basic_cost
    global russian_med_gdp, global_agro_technology, global_building_technology
    while year < 2001:
        print('Year: ' + str(year) + ' Month: ' + str(month))
        if year + month != 1898:
            save_image(regs_dict)
            save_image(regs_dict, type='infra')
            save_image(regs_dict, type='poverty')
            save_image(regs_dict, type='gdp')
            save_image(regs_dict, type='urban')
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        history_sim(regs_dict)
        russian_cumulative_inflation *= russian_monthly_inflation
        if month == 3:
            global_max_med += 0.1
            global_gdp_med *= 1.03
        if month == 9:
            global_max_med += 0.1
            global_agro_technology += 0.1
            global_building_technology += 0.1
        russian_med_gdp = 0.0
        for i in regs_dict.keys():
            russian_med_gdp += regs_dict[i].gdp_per_person
        russian_med_gdp /= len(regs_dict.keys())
        for i in regs_dict.keys():
            regs_dict[i].natural_growth()
            regs_dict[i].economy_growth()
            regs_dict[i].literacy_and_medicine()
            regs_dict[i].urban()
            regs_dict[i].building()

if __name__ == '__main__':
    main()
