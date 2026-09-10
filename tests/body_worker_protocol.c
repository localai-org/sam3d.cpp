/* Exercise the actual private CLI parser without loading any GGUF. */
#define main included_body_infer_main
#include "../apps/body_infer.c"
#undef main
static void test(const unsigned char *data,size_t size,int boundary,int expected){
    FILE *original=stdin;FILE *fixture=tmpfile();
    if(!fixture || fwrite(data,1,size,fixture)!=size)abort();
    rewind(fixture);stdin=fixture;char path[4097];
    int got=read_path(path,boundary);stdin=original;fclose(fixture);
    if(got!=expected){fprintf(stderr,"frame size %zu expected %d got %d\n",size,expected,got);abort();}
    if(got==1 && strlen(path)!=(size_t)data[0]+((size_t)data[1]<<8))abort();
}
int main(void){
    const unsigned char empty[]={0},good[]={3,0,0,0,'a','\n','b'},nul[]={3,0,0,0,'a',0,'b'};
    const unsigned char large[]={1,16,0,0},overflow[]={255,255,255,255},zero[]={0,0,0,0};
    test(empty,0,1,0);test(empty,0,0,-1);
    for(size_t i=1;i<sizeof good;++i)test(good,i,1,-1);
    test(good,sizeof good,1,1);test(nul,sizeof nul,1,-1);
    test(large,sizeof large,1,-1);test(overflow,sizeof overflow,1,-1);test(zero,sizeof zero,1,-1);
    unsigned char maximum[4100];memset(maximum,'x',sizeof maximum);
    maximum[0]=0;maximum[1]=16;maximum[2]=maximum[3]=0;test(maximum,sizeof maximum,1,1);
    puts("Worker framing: boundaries, truncation, NUL, overflow and maximum path passed");return 0;
}
