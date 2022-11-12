/*
    Interpolacao e decimacao recebendo uma amostra por vez.
    Pode aplicar uma distorcao simples atan() depois da interpolacao.
    !!!!!!Implementar soma mais eficiente na interpolação e decimação!!!!!!
    !!!!!!Implementar pipeline no processamento!!!!!!
    !!!!!!Implementar distorção!!!!!!!!
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "sndfile.h"
#include <math.h>
#include <cuda_runtime_api.h> 
#include <time.h> 


__global__ void Interpol(float amostra_in_interpol, float amostra_out_interpol, float *entrada_interpol,float *saida_interpol,float *a_interpol, float *b_interpol, int N_interpol, int N_filtragens)
{    

float soma_entrada=0;
float soma_saida=0;
float entrada_multiplicada[N_interpol+1];
float saida_multiplicada[N_interpol];
int gid = blockIdx.x*blockDim.x+threadIdx.x;

//Reordenando Vetor de entrada para próxima saída (Sem implementação paralela)
if(gid==0)
{
    for(int i=N_interpol+1; i>0; i--)
    {
        entrada_interpol[i]=entrada_interpol[i-1];
    }
    entrada_interpol[0]=amostra_in_interpol;
} 

__syncthreads();

//Calculo da saida (Implementar de forma mais eficiente)
if(gid<N_interpol+1)
{
    entrada_multiplicada[gid]=entrada_interpol[gid]*b_interpol[gid];
}
else
{
    saida_multiplicada[gid]=saida_interpol[gid]*a_interpol[gid];
}

__syncthreads();

if(gid==0)
{
    for(int i=0; i<N_interpol+1)
    {
        soma_entrada=soma_entrada+entrada_multiplicada[i];
    }
}
else
{
    if(gid==1)
    {
        for(int j=0; j<N_interpol)
        {
            soma_saida=soma_saida+saida_multiplicada[j];
        }
    }
}

__syncthreads();

amostra_out_interpol=soma_entrada-soma_saida;

__syncthreads();

//Reordenando Vetor de saida para próxima saída (Sem implementação paralela)
if(gid==0)
{
    for(int i=N_interpol; i>0; i--)
    {
        saida_interpol[i]=saida_interpol[i-1];
    }
    saida_interpol[0]=amostra_out_interpol;
}

__syncthreads();

//Controle dos limites da saída entre +1 e -1
if(gid==0)
{
    if(amostra_out_interpol>1)
    {
        amostra_out_interpol=1;
    }
    else 
    {
        if(amostra_out_interpol<-1)
        {
            amostra_out_interpol=-1;
        }
    }   
}

}

__global__ void Decimacao(float amostra_in_decimacao, float amostra_out_decimacao, float *entrada_decimacao,float *saida_decimacao,float *a_decimacao, float *b_decimacao, int N_decimacao)
{    

float soma_entrada=0;
float soma_saida=0;
float entrada_multiplicada[N_decimacao+1];
float saida_multiplicada[N_decimacao];
int gid = blockIdx.x*blockDim.x+threadIdx.x;

//Reordenando Vetor de entrada para próxima saída (Sem implementação paralela)
if(gid==0)
{
    for(int i=N_decimacao+1; i>0; i--)
    {
        entrada_decimacao[i]=entrada_decimacao[i-1];
    }
    entrada_decimacao[0]=amostra_in_decimacao;
} 

__syncthreads();

//Calculo da saida (Implementar de forma mais eficiente)
if(gid<N_decimacao+1)
{
    entrada_multiplicada[gid]=entrada_decimacao[gid]*b_decimacao[gid];
}
else
{
    saida_multiplicada[gid]=saida_decimacao[gid]*a_decimacao[gid];
}

__syncthreads();

if(gid==0)
{
    for(int i=0; i<N_decimacao+1)
    {
        soma_entrada=soma_entrada+entrada_multiplicada[i];
    }
}
else
{
    if(gid==1)
    {
        for(int j=0; j<N_interpol)
        {
            soma_saida=soma_saida+saida_multiplicada[j];
        }
    }
}

__syncthreads();

amostra_out_decimacao=soma_entrada-soma_saida;

__syncthreads();

//Reordenando Vetor de saida para próxima saída (Sem implementação paralela)
if(gid==0)
{
    for(int i=N_decimacao; i>0; i--)
    {
        saida_decimacao[i]=saida_decimacao[i-1];
    }
    saida_decimacao[0]=amostra_out_decimacao;
}

}


__global__ void Verifica_buffer(float *bufferd, int N)
{
    // printf("Informações sobre os buffer após a primeira inversa: \n");
    printf("buffer[0] = %f , buffer_dt0[1] = %f , buffer_dt0[2] = %f, buffer_dt0[3]= %f, buffer_dt0[4]= %f\n" , bufferd[0], bufferd[1], bufferd[2], bufferd[3], bufferd[4]);
    printf("buffer[N-6] = %f, buffer[N-5] = %f , buffer_dt0[N-4] = %f , buffer_dt0[N-3] = %f, buffer_dt0[N-2]= %f, buffer_dt0[N-1]= %f\n" , bufferd[N-6], bufferd[N-5], bufferd[N-4], bufferd[N-3], bufferd[N-2], bufferd[N-1]);
    printf("---------------------------------------------------------------------------------\n");
}

__global__ void Verifica_Coeficientes(float *coeficientes, int N)
{
    printf("[0] = %f , [1] = %f , [2] = %f, [3]= %f, [4]= %f\n" , coeficientes[0], coeficientes[1], coeficientes[2], coeficientes[3], coeficientes[4]);
    printf("[N-5] = %f , [N-4] = %f , [N-3] = %f, [N-2]= %f, FI[N-1]= %f\n" ,coeficientes[N-5], coeficientes[N-4], coeficientes[N-3], coeficientes[N-2], coeficientes[N-1]);
}

__global__ void Verifica_bufferdL(float *bufferdL, int NL)
{
    // printf("Informações sobre os buffer após a primeira inversa: \n");
    printf("bufferdL[0] = %f , bufferdL[1] = %f , bufferdL[2] = %f, bufferdL[3]= %f, bufferdL[4]= %f\n" , bufferdL[0], bufferdL[1], bufferdL[2], bufferdL[3], bufferdL[4]);
    printf("bufferdL[NL-6] = %f, bufferdL[NL-5] = %f , bufferdL[NL-4] = %f , bufferdL[Nl-3] = %f, bufferdL[NL-2]= %f, bufferdL[NL-1]= %f\n" , bufferdL[NL-6], bufferdL[NL-5], bufferdL[NL-4], bufferdL[NL-3], bufferdL[NL-2], bufferdL[NL-1]);
    printf("---------------------------------------------------------------------------------\n");
}

int main(){

    //Definições do arquivo de leitura e saida
    SNDFILE *file_in,*file_out ;   //Arquivo de entrada e saída
    SF_INFO sfinfo_in,sfinfo_out ; //Arquivo de informações de entrada e saída
    sfinfo_in.format = 0;          //Documentação do libsnd manda fazer isso para arquivos de leitura
    file_in = sf_open ("chinelin_mono_4s.wav", SFM_READ, &sfinfo_in); //Determina o arquivo de entrada
    sfinfo_out=sfinfo_in;
    file_out = sf_open ("chinelin_inter_1_amostra.wav", SFM_WRITE, &sfinfo_out); //Determina o arquivo de saída
    printf("Informações sobre o arquivo de entrada: \n");
    printf("Taxa de amostragem = %d , Frames = % d , Canais = % d \n" , (int) sfinfo_in.samplerate,(int) sfinfo_in.frames,(int)        	  sfinfo_in.channels);                                                              //Mostra algumas caratersiticas do arquivo de entrada 
    
    //Definições do Projeto
    const int L = 10; //Fator de interpolação
    const float grau_distorcao=0.1; //Grau de distorcao
    const int N_filtragens=1; //Numero de filtragens do filtro de interpolação
    const int N_estagios=10; //Numero de estagios de distorcao e filtragem leve
    const int N_interpol=10; //Ordem do filtro de interpolação
    const int N_decimacao=10; //Ordem do filtro de decimacao
    const int N_aliasing=5; //Ordem do filtro leve pós distorção

    //Variaveis da CPU
    float *amostra_in_host; //Amostra para a entrada no processamento feito na GPU
    float *amostra_out_host; //Amostra para a saída no processamento feito na GPU
    
    //Alocação das Variaveis da CPU
    amostra_in_host = (float*)malloc(sizeof(float) * 1);
    amostra_out_host = (float*)malloc(sizeof(float) * 1);

    //Definição dos coeficientes do filtro na CPU
    float a_interpol_host[N_interpol] = {
        #include "a_interpol.txt"
    };
    float b_interpol_host[N_interpol+1] = {
        #include "b_interpol.txt"
    };
    float a_decimacao_host[N_decimacao] = {
        #include "a_decimacao.txt"
    };
    float b_decimacao_host[N_decimacao+1] = {
        #include "b_decimacao.txt"
    };
    float a_aliasing_host[N_aliasing] = {
        #include "a_aliasing.txt"
    };
    float b_aliasing_host[N_aliasing+1] = {
        #include "b_aliasing.txt"
    };

    //Variaveis da GPU
    float *amostra_in_interpol; //Amostra de entrada da interpolação
    float *amostra_out_interpol; //Amostra de saida da interpolação
    float *amostra_in_decimacao; //Amostra de entrada da decimacao
    float *amostra_out_decimacao; //Amostra de saida da decimacao
    const float *a_interpol; //Vetor dos coeficientes recursivos do filtro de interpolação 
    const float *b_interpol; //Vetor dos coeficientes relacionados a entrada do filtro de interpolação
    const float *a_decimacao; //Vetor dos coeficientes recursivos do filtro de decimacao
    const float *b_decimacao; //Vetor dos coeficientes relacionados a entrada do filtro de decimação
    const float *a_aliasing; //Vetor dos coeficientes recursivos do filtro pós distorção 
    const float *b_aliasing; //Vetor dos coeficientes relacionados a entrada do filtro pós distorção
    float *entrada_interpol; //Vetor de entrada na interpolação
    float *saida_interpol; //Vetor de saída na interpolação
    float *entrada_decimacao; //Vetor de entrada na decimação
    float *saida_decimacao; //Vetor de saída na decimação
    float *entrada_aliasing; //Vetor de entrada no filtro pós distorção
    float *saida_aliasing; //Vetor de saída no filtro pós distorção

    //Alocação das Variaveis da GPU
    cudaMalloc((void**)&amostra_in_interpol, sizeof(float) * 1);
    cudaMalloc((void**)&amostra_out_interpol, sizeof(float) * 1);
    cudaMalloc((void**)&amostra_in_decimacao, sizeof(float) * 1);
    cudaMalloc((void**)&amostra_out_decimacao, sizeof(float) * 1);  
    cudaMalloc((void**)&a_interpol, sizeof(float) * N_interpol);
    cudaMalloc((void**)&b_interpol, sizeof(float) * N_interpol+1);
    cudaMalloc((void**)&a_decimacao, sizeof(float) * N_decimacao);
    cudaMalloc((void**)&b_decimacao, sizeof(float) * N_decimacao+1);
    cudaMalloc((void**)&a_aliasing, sizeof(float) * N_aliasing);
    cudaMalloc((void**)&b_aliasing, sizeof(float) * N_aliasing+1);
    cudaMalloc((void**)&saida_interpol, sizeof(float) * N_interpol);
    cudaMalloc((void**)&entrada_interpol, sizeof(float) * N_interpol+1);
    cudaMalloc((void**)&saida_decimacao, sizeof(float) * N_decimacao);
    cudaMalloc((void**)&entrada_decimacao, sizeof(float) * N_decimacao+1);
    cudaMalloc((void**)&entrada_aliasing, sizeof(float) * N_aliasing);
    cudaMalloc((void**)&saida_aliasing, sizeof(float) * N_aliasing+1);

    //Definição dos coeficientes dos filtros na GPU;
    cudaMemcpy(a_interpol, a_interpol_host, sizeof(float)*N_interpol, cudaMemcpyHostToDevice);
    cudaMemcpy(b_interpol, b_interpol_host, sizeof(float)*N_interpol+1, cudaMemcpyHostToDevice);
    cudaMemcpy(a_decimacao, a_decimacao_host, sizeof(float)*N_decimacao, cudaMemcpyHostToDevice);
    cudaMemcpy(b_decimacao, b_decimacao_host, sizeof(float)*N_decimacao+1, cudaMemcpyHostToDevice);
    cudaMemcpy(a_aliasing, a_aliasing_host, sizeof(float)*N_aliasing, cudaMemcpyHostToDevice);
    cudaMemcpy(b_aliasing, b_aliasing_host, sizeof(float)*N_aliasing+1, cudaMemcpyHostToDevice);

    //Condições iniciais nulas
    cudaMemset((void**)&saida_interpol, 0, sizeof(float)*N_interpol); 
    cudaMemset((void**)&entrada_interpol, 0, sizeof(float)*N_interpol+1); 
    cudaMemset((void**)&saida_decimacao, 0, sizeof(float)*N_decimacao);
    cudaMemset((void**)&entrada_decimacao, 0, sizeof(float)*N_decimacao+1);
    cudaMemset((void**)&saida_aliasing, 0, sizeof(float)*N_aliasing);
    cudaMemset((void**)&entrada_aliasing, 0, sizeof(float)*N_aliasing+1);

    //Variaveis para controle de streams e tempo
    cudaEvent_t inicio_interpolacao, fim_interpolacao,inicio_distorcao, fim_distorcao, inicio_decimacao,fim_decimacao;
    cudaEventCreate(&inicio_interpolacao);
    cudaEventCreate(&fim_interpolacao);
    cudaEventCreate(&inicio_distorcao);
    cudaEventCreate(&fim_distorcao);
    cudaEventCreate(&inicio_decimacao);
    cudaEventCreate(&fim_decimacao);
    clock_t inicio_total;
	clock_t fim_total;

    //Prints iniciais para Verificação dos filtros
    printf("Printando a_interpol: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(a_interpol, N_interpol);
    printf("\n");
    printf("Printando b_interpol: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(b_interpol, N_interpol+1);
    printf("\n");
    printf("---------------------------------------------------------------------------------\n");
    printf("Printando a_decimacao: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(a_decimacao, N_decimacao);
    printf("\n");
    printf("Printando b_decimacao: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(b_decimacao, N_decimacao+1);
    printf("\n");
    printf("---------------------------------------------------------------------------------\n");
    printf("Printando a_aliasing: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(a_aliasing, N_aliasing);
    printf("\n");
    printf("Printando b_aliasing: \n");
    Verifica_Coeficientes<<< 1, 1 >>>(b_aliasing, N_aliasing+1);
    printf("\n");
    printf("---------------------------------------------------------------------------------\n");

    //Leitura de amostras e processamento
    int read_count;
    int num_iteracoes = 0;

    double soma_total = 0;
    float soma_interpolacao = 0;
    float soma_decimacao = 0;

    float tempo_interpolacao;
    float tempo_decimacao;

    float max_entrada = 0;
    float max_saida = 0;

    while (num_iteracoes<176450 && (read_count = (int) sf_read_float (file_in, amostra_in_host, 1)))
    {
        inicio_total = clock();

        // Copiando o buffer de entrada para o device
        cudaMemcpy(amostra_in_interpol, amostra_in_host, sizeof(float)*1, cudaMemcpyHostToDevice);

        //Controle para receber L-1 amostras iguais depois de cada amostra (Precisa trocar por uma forma de pipeline)
        for(int i=0; i<L; i++)
        {
            //Interpolação
            cudaEventRecord(inicio_interpolacao);
            Interpol<<<1,N_interpol+1>>>(amostra_in_interpol, amostra_out_interpol, entrada_interpol, saida_interpol, a_interpol, b_interpol, N_interpol, N_filtragens);
            cudaEventRecord(fim_interpolacao);

            //Estágios de Distorção e filtragem (Implementar depois do Pipeline funcionar)
            //
            //
            //

            //Decimação
            cudaEventRecord(inicio_decimacao);
            Decimacao<<<1,2*N_decimacao+1>>>(amostra_out_interpolacao, amostra_out_decimacao, entrada_decimacao, saida_decimacao, a_decimacao, b_decimacao, N_decimacao)
            cudaEventRecord(fim_decimacao);

            if(i==0)
            {
                cudaMemcpy(amostra_out_host, amostra_out_decimacao, sizeof(float)*1, cudaMemcpyDeviceToHost);
            }
        }     

        cudaDeviceSynchronize();

        fim_total = clock();

        soma_total += ((double)(fim_total - inicio_total)) / CLOCKS_PER_SEC;
        
        cudaEventElapsedTime(&tempo_interpolacao, inicio_interpolacao, fim_interpolacao);
        soma_interpolacao += tempo_interpolacao/1000;

        cudaEventElapsedTime(&tempo_decimacao, inicio_decimacao, fim_decimacao);
        soma_decimacao += tempo_decimacao/1000;

        
        // Escrevendo no arquivo de saida
        sf_write_float (file_out, amostra_out_host, read_count) ;

        num_iteracoes++;
    }

    printf("Acabou\n");

    printf("max_antes: %f\n", max_entrada);
    printf("max_depois: %f\n\n", max_saida);

    printf("Nro de blocos processados.....: %d\n", num_iteracoes); 
    printf("Media do tempo total..........: %f\n", soma_total/num_iteracoes);
    printf("Media do tempo interpolacao...: %f\n", soma_interpolacao/num_iteracoes);
    printf("Media do tempo decimacao......: %f\n", soma_decimacao/num_iteracoes);

    //Liberando a memória alocada
    free(amostra_in_host);
    free(amostra_out_host);
    free(a_interpol_host);
    free(b_interpol_host);
    free(a_decimacao_host);
    free(b_decimacao_host);
    free(a_aliasing_host);
    free(a_aliasing_host);

    cudaFree(amostra_in_interpol);
    cudaFree(amostra_out_interpol);
    cudaFree(amostra_in_decimacao);
    cudaFree(amostra_out_decimacao);   
    cudaFree(a_interpol);
    cudaFree(b_interpol);
    cudaFree(a_decimacao);
    cudaFree(b_decimacao);
    cudaFree(a_aliasing);
    cudaFree(a_aliasing);
    cudaFree(entrada_interpol);
    cudaFree(saida_interpol);
    cudaFree(entrada_decimacao);
    cudaFree(saida_decimacao);
    cudaFree(entrada_aliasing);
    cudaFree(saida_aliasing);

    cudaEventDestroy(inicio_interpolacao);
    cudaEventDestroy(fim_interpolacao);
    cudaEventDestroy(inicio_decimacao);
    cudaEventDestroy(fim_decimacao);

}