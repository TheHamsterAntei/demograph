import numpy as np
import sys
import pandas as pd
import os
from PIL import Image


#Глобальные параметры (Россия и мир):
#Технические:
img = Image.open('Source.png').convert('RGB')
img_blue_colors = [0 for i in range(0, 256)]
for i in range(0, img.width):
    for j in range(0, img.height):
        t = img.getpixel((i, j))
        if t[0] == 0 and t[1] == 0:
            img_blue_colors[t[2]] += 1
gif_russians = []
gif_density = []
gif_gdp = []
gif_towners = []
#Мир:
year = 1897
month = 1
global_max_med = 10.0 #Максимальный уровень медицины в мире
global_gdp_med = 300.0 #Средний ВВП на душу населения в развитых странах
global_stability = 1.0 #Мировая стабильность
global_openness = 1.0 #Мировая открытость
#Россия:
russian_toleracy = 0.2 #Терпимость к меньшинствам
russian_openness = 1.0 #Открытость границ
russian_stability = 0.8 #Стабильность в стране
russian_democracy = 0.0 #Демократичность режима
russian_selfcon = 0.6 #Национальное самосознание русских
russian_humilation = 0.1 #Степень унижения русских/России


class Population:
    def __init__(self, region, nations, row):
        self.pop_by_nations = {}
        self.rel_by_nations = {}
        self.region = region
        self.nations = nations
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
            for k in range(1, 80):
                age_weight.append(age_weight[k - 1] * (0.997 - 0.002 * k - 0.01 * max(0, k - 40)
                                                       - 0.02 * max(0, k - 60)))
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
        '''Низкая стабильность и нахождение в состоянии войны снижают желание людей заводить детей'''
        region_isrussian = self.region.isrussian
        if region_isrussian == 1:
            region_stability *= (russian_stability**0.5)
        '''Должны ли применяться к региону модификаторы нахождения в составе России'''
        region_prosperity = self.region.person_prosperity
        '''Достаток семей. Желание иметь детей наивысшее у людей с низким уровнем дохода, однако этот параметр имеет
        самую низкую важность'''

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
                #Шанс умереть от несчастного случая наивысший в подростковом возрасте и в молодости
                #Также этот эффект отражает шанс умереть младенцем
                basic_risk = 0.0
                if k == 0:
                    basic_risk = 0.012 - max(0.0115, 0.0001 * medicine)
                if 12 >= k > 0:
                    basic_risk = 0.00009
                if 28 > k > 12:
                    basic_risk = 0.00014
                if k >= 28:
                    basic_risk = 0.00009
                #Эффекты стабильности
                stability_risk = 0.00001 + 0.001 * (1 - region_stability)
                #Эффекты войны
                war_risk = max(0.0005 * region_iswar, 0.012 * region_iswar * int(bool(45 >= k >= 18)))
                men_dec = men / 12
                men_inc = (men / 12) * (1 - elder_risk - basic_risk - stability_risk - war_risk)
                if men_dec < 1 and men > 0:
                    men_dec = np.random.choice([0, 1], p=(1 - men_dec, men_dec))
                if int(men_dec) == int(men_inc):
                    men_inc -= np.random.choice([0, 1], p=(1 - elder_risk - basic_risk - stability_risk - war_risk,
                                                           elder_risk + basic_risk + stability_risk + war_risk))
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
                    basic_risk = 0.012 - max(0.0115, 0.0001 * medicine)
                if 12 >= k > 0:
                    basic_risk = 0.00009
                if 28 > k > 12:
                    basic_risk = 0.00011
                if k >= 28:
                    basic_risk = 0.00006
                # Эффекты стабильности
                stability_risk = 0.00001 + 0.001 * (1 - region_stability)
                # Эффекты войны
                war_risk = 0.0005
                women_dec = int(women / 12)
                women_inc = int((women / 12) * (1 - elder_risk - basic_risk - stability_risk - war_risk))
                if women_dec < 1 and women > 0:
                    women_dec = np.random.choice([0, 1], p=(1 - women_dec, women_dec))
                if int(women_dec) == int(women_inc):
                    women_inc -= np.random.choice([0, 1], p=(1 - elder_risk - basic_risk - stability_risk - war_risk,
                                                           elder_risk + basic_risk + stability_risk + war_risk))
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
                    preg_risk = preg_risk = 0.1 - min(0.095, 0.006 * medicine)
                if 40 <= k:
                    basic_chance = 0.001
                    preg_risk = preg_risk = 0.2 - min(0.19, 0.006 * medicine)
                if region_iswar == 1:
                    basic_chance *= 0.5
                basic_chance *= k_chance
                basic_chance *= (region_stability + 0.3)**0.8
                basic_chance *= 4 - (3.8 * (real_dt**0.3))
                if region_prosperity < -100.0:
                    basic_chance *= 2
                if -100.0 <= region_prosperity < -50.0:
                    basic_chance *= 1.5
                if -50.0 <= region_prosperity < 0:
                    basic_chance *= 1.2
                birth_count = self.pop_by_nations[i][k][1] * basic_chance
                if birth_count < 0:
                    birth_count = 0
                if 0.0 < birth_count < 1.0:
                    birth_count = np.random.choice([0, 1], p=(1 - birth_count, birth_count))
                if birth_count >= 2.0:
                    self.pop_by_nations[i][0][0] += int(birth_count / 2)
                    self.pop_by_nations[i][0][1] += int(birth_count - int(birth_count / 2))
                if birth_count == 1.0:
                    if np.random.choice([0, 1], p=(0.5, 0.5)) == 0:
                        self.pop_by_nations[i][0][0] += 1
                    else:
                        self.pop_by_nations[i][0][1] += 1
        #Шаг 3. Пересчитать население
        men_pop = sum([sum([self.pop_by_nations[k1][k2][0] for k2 in range(0, len(self.pop_by_nations[k1]))])
                          for k1 in self.pop_by_nations.keys()])
        women_pop = sum([sum([self.pop_by_nations[k1][k2][1] for k2 in range(0, len(self.pop_by_nations[k1]))])
                          for k1 in self.pop_by_nations.keys()])
        population = men_pop + women_pop
        self.region.population = population
        self.region.men_pop = men_pop
        self.region.women_pop = women_pop


class Region:
    def __init__(self, nations, row):
        #Неизменяемые параметры:
        self.name = row[0]
        self.img_color = row[len(row) - 1]
        self.square = img_blue_colors[self.img_color]

        #Экономические параметры:
        self.gdp_per_person = row[1]
        self.region_gdp = row[2]
        self.person_prosperity = row[3]
        self.life_cost = row[4]
        self.stratification = row[5] * 0.01
        self.infrastructure = row[6]
        self.housing = 1

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

        #Политические параметры:
        self.isrussian = 1
        self.iswar = 0 #Находится ли регион в составе государства, ведущего войну
        self.openness = 0.6
        self.governor_eff = 1
        self.stability = 1
        self.monarch_power = 0.5 #Поддержка монархии
        self.autocrat_power = 0.5 #Поддержка авторитаризма
        self.democracy_power = 0.3 #Поддержка демократии
        self.natpop_power = 0.0  #Поддержка национал-популизма
        self.liberal_power = 0.4 #Поддержка либерализма
        self.socialist_power = 0.5 #Поддержка социализма
        self.communist_power = 0.1 #Поддержка коммунизма
        self.conserv_power = 0.7 #Поддержка традиционных ценностей
        self.progress_power = 0.2 #Поддержка прогрессивных ценностей

        #Природные параметры:
        self.climate = row[10]

    def natural_growth(self):
        self.population_object.natural_growth()
        if self.dem_transition_towns < 1:
            self.dem_transition_towns += 0.001
        if self.dem_transition_towns > 0.8:
            if self.dem_transition_rural < 1:
                self.dem_transition_rural += 0.0005


def save_density_image(regs_dict):
    dict_blues = {}
    max_dens = 1.0
    for i in regs_dict.keys():
        dict_blues[regs_dict[i].img_color] = i
        max_dens = max(max_dens, regs_dict[i].pop_density)
    img_density = img.copy()
    for x in range(0, img_density.width):
        for y in range(0, img_density.height):
            t = img_density.getpixel((x, y))
            if t[0] == 0 and t[1] == 0:
                if t[2] in dict_blues.keys():
                    density = (regs_dict[dict_blues[t[2]]].pop_density) ** 0.5 / (max_dens) ** 0.5 * 255
                    img_density.putpixel((x, y), (int(density * 0.8), 0, int(density * 0.4)))
    img_density.save("Output//Density//Dens" + str(int(((year - 1897) * 12 + month - 1) / 6)) + '.png', 'PNG')

def main():
    data = pd.read_excel('RegData.xlsx').drop(['Итого', 'Столица субъекта'], axis=1)
    if not os.path.exists('Output\\'):
        os.mkdir('Output')
    if not os.path.exists('Output\\Density\\'):
        os.mkdir('Output\\Density')
    nations = data.columns.to_list()
    nations = nations[15:len(nations) - 1]
    regs_dict = {}
    for i in data.iterrows():
        row = i[1].to_list()
        regs_dict[row[0]] = Region(nations, row)
    global month, year
    while year < 2001:
        if ((year - 1897) * 12 + month - 1) % 6 == 0:
            save_density_image(regs_dict)
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        for i in regs_dict.keys():
            regs_dict[i].natural_growth()
    #Ниже тесты рисования
    sys.exit(0)
    dict_blues = {}
    for i in regs_dict.keys():
        dict_blues[regs_dict[i].img_color] = i
        max_dens = max(max_dens, regs_dict[i].pop_density)
        max_gdp = max(max_gdp, regs_dict[i].region_gdp)
    img_density = img.copy()
    for x in range(0, img_density.width):
        for y in range(0, img_density.height):
            t = img_density.getpixel((x, y))
            if t[0] == 0 and t[1] == 0:
                if t[2] in dict_blues.keys():
                    density = (regs_dict[dict_blues[t[2]]].pop_density)**0.5 / (max_dens)**0.5 * 255
                    img_density.putpixel((x, y), (int(density * 0.8), 0, int(density * 0.4)))
    img_density.save("Output//PopDens.png", 'PNG')
    img_gdp = img.copy()
    for x in range(0, img_gdp.width):
        for y in range(0, img_gdp.height):
            t = img_gdp.getpixel((x, y))
            if t[0] == 0 and t[1] == 0:
                if t[2] in dict_blues.keys():
                    gdp = (regs_dict[dict_blues[t[2]]].region_gdp)**0.5 / (max_gdp)**0.5 * 255
                    img_gdp.putpixel((x, y), (int(gdp * 0.8), 0, int(gdp * 0.4)))
    img_gdp.save("Output//GDP.png", 'PNG')
    img_russians = img.copy()
    for x in range(0, img_russians.width):
        for y in range(0, img_russians.height):
            t = img_russians.getpixel((x, y))
            if t[0] == 0 and t[1] == 0:
                if t[2] in dict_blues.keys():
                    russians = (regs_dict[dict_blues[t[2]]].population_object.pop_by_nations['Русские'] /
                                regs_dict[dict_blues[t[2]]].population * 255)
                    img_russians.putpixel((x, y), (int(russians * 0.8), 0, int(russians * 0.4)))
    img_russians.save("Output//Russians.png", 'PNG')

if __name__ == '__main__':
    main()
