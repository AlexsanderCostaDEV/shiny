require(extRemes)
require(geosphere)
require(randomForest)
require(e1071)
require(ggplot2)
require(ggpubr)
require(rgdal)
require(sf)
require(ggsn)

## leitura dos dados em formato csv
dados <- read.csv2("C:/Users/alexs/Desktop/tcc/dados.csv")
str(dados)

## Nesse trecho, está sendo definida uma função chamada rl que recebe um vetor x como parâmetro. Essa função utiliza o método de ajuste 
# da distribuição Gumbel (fevd()) para ajustar um modelo aos dados x. Em seguida, a função return.level() é utilizada para calcular o nível de retorno, 
# nesse caso, com um período de retorno de 10 anos. O resultado é convertido para um valor numérico através da função as.numeric(). Essa função permite 
# calcular o valor do nível de retorno para um dado conjunto de dados.
rl <- function(x){
  gumbel_fit <- fevd(x, type="Gumbel")
  as.numeric(return.level(gumbel_fit, return.period = 10))
}

## Nesse trecho, está sendo utilizado a função aggregate() para realizar a agregação dos dados. Está sendo calculado o valor do nível de retorno 
# de 10 anos (rl) para a variável PrecMax em relação às variáveis Meso, Posto, Lat, Long e DistL. A agregação é feita utilizando os dados do objeto 
# dados. O resultado é armazenado no objeto rl10Anos.
rl10Anos <- aggregate(PrecMax ~ Meso + Posto + Lat + Long + DistL,
                      data=dados,
                      FUN = rl)   


## Coordenados Ponto Seixas
lat1 <- -7.148671
long1 <- -34.796150

## Coordenadas de Lat e Long presentes em rl10anos são atribuidas as variáveis
lat2 <- rl10Anos$Lat
long2 <- rl10Anos$Long

## Nesse trecho, está sendo calculada a distância entre o ponto de referência definido pelas coordenadas lat1 e long1 e os pontos representados 
# pelas coordenadas lat2 e long2 presentes no objeto rl10Anos. O resultado é dividido por 1000 para converter a unidade para quilômetros. 
dp <- distm(c(long1,lat1), cbind(long2, lat2),
            fun = distHaversine)/1000

## Nesse trecho, a distância calculada (dp) é adicionada como uma nova coluna chamada dist no objeto rl10Anos. Em seguida, é realizada uma análise
# de correlação entre todas as colunas do objeto rl10Anos, exceto as duas primeiras colunas (cor(rl10Anos[,-c(1,2)])). Por fim, é criado um 
# gráfico (plot) dos dados presentes no objeto rl10Anos, excluindo as duas primeiras colunas (plot(rl10Anos[,-c(1,2)])).
rl10Anos$dist <- as.numeric(dp)
cor(rl10Anos[,-c(1,2)])
plot(rl10Anos[,-c(1,2)])

## resumo estatistico da variável
summary(rl10Anos)

## Nesse trecho, uma amostra aleatória de tamanho 110 é selecionada do intervalo de 1 a 123 (sample(1:123, size = 110)) e armazenada na variável np. 
# Em seguida, são criados dois conjuntos de dados: train e test. O conjunto train contém as linhas correspondentes aos índices presentes em np, ou seja, 
# são selecionadas aleatoriamente 110 linhas do objeto rl10Anos. Já o conjunto test contém as linhas que não estão presentes em np, ou seja, são 
# selecionadas as linhas restantes do objeto rl10Anos. Essa divisão é comumente utilizada para separar um conjunto de dados em um conjunto de treinamento 
# e um conjunto de teste para avaliar a performance do modelo.
np <- sample(1:123, size = 110)
train <- rl10Anos[np,]
test <- rl10Anos[-np,]


## Machine Learning
## Nesse trecho, está sendo realizado o ajuste de um modelo SVM (Support Vector Machine) utilizando a função tune(). O objetivo é otimizar os parâmetros 
# do modelo. A variável de resposta é PrecMax e as variáveis preditoras são Lat, Long e dist presentes no conjunto de dados rl10Anos. A otimização é 
# realizada para os parâmetros epsilon no intervalo de 0 a 0.20 com incremento de 0.01 e cost variando de 1 a 100. A função tune() busca encontrar a 
# combinação ideal de parâmetros que resulta no melhor desempenho do modelo.
OptModelsvm=tune(svm, PrecMax ~ Lat + Long + dist,
                 data=rl10Anos,
                 ranges=list(epsilon=seq(0,0.20,0.01), cost=1:100))

## Nesse trecho, a variável BstModel recebe o modelo com os melhores parâmetros encontrados durante a otimização. Através do comando BstModel, é 
# exibido o modelo com os valores ótimos de parâmetros encontrados.
BstModel=OptModelsvm$best.model
BstModel


## Nesse trecho, está sendo realizado o ajuste de um modelo SVM com parâmetros fixos. A função tune() é utilizada novamente, mas dessa vez os parâmetros 
# epsilon e cost são definidos como valores fixos, respectivamente, 0.10 e 36. O parâmetro gamma varia de 0.1 a 10 com incremento de 0.1. O modelo será 
# ajustado utilizando as mesmas variáveis preditoras e conjunto de dados rl10Anos.
OptModelsvm = tune(svm, PrecMax ~ Lat + Long + dist,
                 data=rl10Anos,
                 ranges=list(epsilon=0.10, cost = 36, gamma = seq(0.1,10,0.1)))

## Nesse trecho, a variável BstModel recebe o modelo com os melhores parâmetros encontrados durante a otimização. O comando BstModel é utilizado para 
# exibir o modelo com os valores ótimos de parâmetros encontrados.
BstModel = OptModelsvm$best.model
BstModel

## Nesse trecho, a função predict() é utilizada para fazer previsões utilizando o modelo ajustado com os melhores parâmetros (BstModel). As previsões 
# são armazenadas na coluna PredYBst do objeto rl10Anos.
rl10Anos$PredYBst = predict(BstModel,rl10Anos)

## Nesse trecho, o coeficiente de determinação R² é calculado utilizando a função cor(). É medida a correlação entre as variáveis PredYBst (previsões 
# feitas pelo modelo) e PrecMax (variável de resposta original) presentes no objeto rl10Anos. O coeficiente de determinação R² é obtido elevando ao 
# quadrado a correlação calculada.
cor(rl10Anos$PredYBst,rl10Anos$PrecMax)^2


## Nesse trecho, está sendo criado um gráfico de dispersão utilizando a função ggplot() do pacote ggplot2. As variáveis PredYBst e PrecMax são mapeadas 
# nos eixos x e y, respectivamente. geom_point() é usado para exibir os pontos no gráfico. stat_regline_equation() é utilizado para adicionar uma linha 
# de regressão ao gráfico, juntamente com a equação da linha e o coeficiente de determinação R². A fórmula da regressão é definida como formula <- y ~ x.
formula <- y ~ x
ggplot(rl10Anos, aes(x = PredYBst, y = PrecMax))+
  geom_point() +
  stat_regline_equation(
    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~")),
    formula = formula
  )


## Leitura e plot do arquivo .shp do mapa da Paraíba
ctba <- readOGR("C:/Users/alexs/Desktop/tcc/25MEE250GC_SIR.shp")
plot(ctba)

## Adicionando pontos vermelhos no mapa anteriormente plotado. 
points(rl10Anos$Long, rl10Anos$Lat, col='red')
np=56585


## Nesse trecho, está sendo realizada uma amostragem de pontos regularmente espaçados no objeto ctba, utilizando a função spsample(). Os pontos são 
# amostrados no objeto ctba em uma quantidade determinada pelo valor da variável np (56585), e o tipo de amostragem é definido como "regular".
ptsreg <- spsample(ctba, np, type = "regular") 


## Nesse trecho, um objeto ddp é criado com as coordenadas dos pontos amostrados. As coordenadas de longitude são armazenadas na coluna 'Long' e as 
# coordenadas de latitude são armazenadas na coluna 'Lat'.
ddp <- data.frame('Long' = ptsreg@coords[,1],
                  'Lat'  = ptsreg@coords[,2])

## Nesse trecho, são adicionados os pontos no gráfico existente. Os pontos são definidos pelas coordenadas de longitude (ddp$Long) e latitude (ddp$Lat) 
# do objeto ddp. O parâmetro cex=0.050 controla o tamanho dos pontos no gráfico.
points(x = ddp$Long, y = ddp$Lat, cex=0.050)

long2 <- ddp$Long
lat2 <- ddp$Lat

## Nesse trecho, está sendo calculada a distância entre os pontos amostrados (coordenadas de longitude e latitude armazenadas em long2 e lat2, 
# respectivamente) e o ponto de referência (coordenadas de longitude long1 e latitude lat1). A função distm() é utilizada com a fórmula de Haversine 
# (fun = distHaversine). O resultado é dividido por 1000 para converter a unidade para quilômetros.
dp <- distm(c(long1,lat1), cbind(long2, lat2),
            fun = distHaversine)/1000

## Nesse trecho, duas colunas são adicionadas ao objeto ddp. A coluna dist recebe o valor numérico da distância calculada (dp) e a coluna PredSVM 
# recebe as previsões feitas pelo modelo BstModel para as coordenadas presentes no objeto ddp.
ddp$dist <- as.numeric(dp)
ddp$PredSVM <- predict(BstModel, ddp)
summary(ddp$PredSVM)

cores <- c("#ff0000", "#ff8c00", "#ffff00", "#aaff00", "#00ff7f", "#0000ff")


ctba1 <- st_read("C:/Users/alexs/Desktop/tcc/25MEE250GC_SIR.shp", crs = "NAD27")


## Nesse trecho, está sendo criado um gráfico utilizando a biblioteca ggplot2. São adicionados pontos no gráfico com base nas coordenadas de latitude 
# (Lat) e longitude (Long) do objeto ddp. A cor dos pontos é definida pela variável PredSVM. Além disso, são adicionadas linhas de limite do objeto 
# ctba1 (shapefile) e uma escala de cores. O gráfico também inclui uma escala de distância, rótulos de eixos, ajustes de tema e outras configurações 
# estéticas relacionadas à aparência do gráfico.
ggplot(ddp) +
  geom_point(aes(y = Lat, x = Long, colour = PredSVM), size=0.05) +
  geom_sf(data=ctba1,  fill=NA, size=.15, show.legend = F) +
  scale_colour_stepsn(colours = cores, 
                      limits = c(80, 180),
                      guide = guide_coloursteps(even.steps = TRUE,
                                                show.limits = TRUE),
                      breaks = seq(80,180,20))+
  scalebar(ctba1, dist = 50, dist_unit = "km", location = "bottomleft", height = 0.03,
           transform = TRUE, model = "WGS84", st.size = 1.5, st.dist = 0.06)+
  labs(color = "10 anos ", x = "Longitude", y = "Latitude") +
  theme_bw(base_size = 11, base_family = 'serif') +
  theme(legend.position="bottom",
        strip.text.x=element_text(angle=00, hjust=0.25, vjust=0.5)) +
  theme(legend.key.width=unit(1.5, "cm"), legend.key.height=unit(0.2, "cm"))
