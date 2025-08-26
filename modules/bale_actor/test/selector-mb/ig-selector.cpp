#include <shmem.h>
extern "C" {
#include <spmat.h>
}
#include "selector-mb.h"

#define THREADS   shmem_n_pes()
#define MYTHREAD  shmem_my_pe()

constexpr std::size_t MB_REQUEST  = 0;
constexpr std::size_t MB_RESPONSE = 1;

struct Resp2x64 {
    int64_t idx;
    int64_t val;
};

static inline uint64_t pack_req(uint64_t req_idx, uint64_t local_idx) {
    return (req_idx << 32) | (local_idx & 0xffffffffULL);
}
static inline void unpack_req(uint64_t packed, uint64_t &req_idx, uint64_t &local_idx) {
    req_idx   = (packed >> 32);
    local_idx = (packed & 0xffffffffULL);
}

/*!
 * \brief Selector variant of indexgather using variadic mailboxes.
 * \param tgt       target array for gathered values (length l_num_req)
 * \param pckindx   packed (local_index<<16 | pe) for each request (length l_num_req)
 * \param l_num_req number of requests per PE
 * \param ltable    local partition of the distributed table (length ltab_siz)
 * \return average runtime across PEs
 */
double ig_selector(int64_t *tgt, int64_t *pckindx, int64_t l_num_req, int64_t *ltable) {
    using IGSel = hclib::Selector<BUFFER_SIZE, uint64_t, Resp2x64>; // mb0: uint64_t, mb1: Resp2x64
    minavgmaxD_t stat[1];

    IGSel sel(false);
    sel.template mailbox<MB_REQUEST>().process = [ltable, &sel](uint64_t packed, int sender_rank) {
        uint64_t req_idx_u = 0, local_idx_u = 0;
        unpack_req(packed, req_idx_u, local_idx_u);

        Resp2x64 r;
        r.idx = static_cast<int64_t>(req_idx_u);
        r.val = ltable[static_cast<size_t>(local_idx_u)];

        sel.template send<MB_RESPONSE>(r, sender_rank);
    };

    sel.template mailbox<MB_RESPONSE>().process = [tgt](Resp2x64 r, int) {
        tgt[r.idx] = r.val;
    };

    lgp_barrier();
    double tm = wall_seconds();

    hclib::finish([&]() {
        sel.start();
        for (int64_t i = 0; i < l_num_req; i++) {
            int64_t lindx = (pckindx[i] >> 16);              // local index at destination PE
            int      dest = static_cast<int>(pckindx[i] & 0xffff); // destination PE
            uint64_t req  = pack_req(static_cast<uint64_t>(i),
                                     static_cast<uint64_t>(lindx));
            sel.template send<MB_REQUEST>(req, dest);
        }
        sel.template done<MB_REQUEST>();
    });

    tm = wall_seconds() - tm;
    lgp_barrier();
    lgp_min_avg_max_d(stat, tm, THREADS);
    return stat->avg;
}

int64_t ig_check_and_zero(int64_t use_model, int64_t *tgt, int64_t *index, int64_t l_num_req) {
    int64_t errors = 0;
    lgp_barrier();
    for (int64_t i = 0; i < l_num_req; i++) {
        if (tgt[i] != (-1) * (index[i] + 1)) {
            errors++;
            if (errors < 5) {
                fprintf(stderr,"ERROR: model %ld: Thread %d: tgt[%ld] = %ld != %ld)\n",
                        use_model, MYTHREAD, i, tgt[i], (-1)*(index[i] + 1));
            }
        }
        tgt[i] = 0;
    }
    if (errors > 0)
        fprintf(stderr,"ERROR: %ld: %ld total errors on thread %d\n", use_model, errors, MYTHREAD);
    lgp_barrier();
    return errors;
}

int main(int argc, char * argv[]) {

  const char *deps[] = { "system", "bale_actor" };
  hclib::launch(deps, 2, [=] {

    int64_t i;
    int64_t buf_cnt = 1024;
    int64_t models_mask = 0;
    int64_t ltab_siz = 100000;
    int64_t l_num_req  = 1000000;
    int64_t cores_per_node = 0;
    int64_t num_errors = 0L, total_errors = 0L;
    int64_t printhelp = 0;

    int opt;
    while ((opt = getopt(argc, argv, "hb:M:n:c:T:")) != -1) {
      switch(opt) {
      case 'h': printhelp = 1; break;
      case 'b': sscanf(optarg,"%ld" ,&buf_cnt);   break;
      case 'M': sscanf(optarg,"%ld" ,&models_mask);  break;
      case 'n': sscanf(optarg,"%ld" ,&l_num_req);   break;
      case 'T': sscanf(optarg,"%ld" ,&ltab_siz);   break;
      case 'c': sscanf(optarg,"%ld" ,&cores_per_node); break;
      default:  break;
      }
    }

    T0_fprintf(stderr,"Running ig on %d threads\n", THREADS);
    T0_fprintf(stderr,"buf_cnt (number of buffer pkgs)      (-b)= %ld\n", buf_cnt);
    T0_fprintf(stderr,"Number of Request / thread           (-n)= %ld\n", l_num_req );
    T0_fprintf(stderr,"Table size / thread                  (-T)= %ld\n", ltab_siz);
    T0_fprintf(stderr,"models_mask                          (-M)= %ld\n", models_mask);
    T0_fprintf(stderr,"models_mask is or of 1,2,4,8,16 for agi,exstack,exstack2,conveyor,alternate)\n");

    // Allocate and populate the shared table array
    int64_t tab_siz = ltab_siz * THREADS;
    int64_t * table  = (int64_t*)lgp_all_alloc(tab_siz, sizeof(int64_t)); assert(table != NULL);
    int64_t * ltable = lgp_local_part(int64_t, table);

    // Fill table with negative of global index for checking
    for (i=0; i<ltab_siz; i++)
      ltable[i] = (-1)*(i*THREADS + MYTHREAD + 1);

    int64_t *index   = (int64_t*)calloc(l_num_req, sizeof(int64_t)); assert(index != NULL);
    int64_t *pckindx = (int64_t*)calloc(l_num_req, sizeof(int64_t)); assert(pckindx != NULL);

    int64_t indx, lindx, pe;
    srand(MYTHREAD + 5);
    for (i = 0; i < l_num_req; i++) {
      indx = rand() % tab_siz;
      index[i] = indx;
      lindx = indx / THREADS;      // local index at destination PE
      pe  = indx % THREADS;        // destination PE
      pckindx[i] = (lindx << 16) | (pe & 0xffff);
    }

    int64_t *tgt  = (int64_t*)calloc(l_num_req, sizeof(int64_t)); assert(tgt != NULL);

    lgp_barrier();

    double laptime = ig_selector(tgt, pckindx, l_num_req, ltable);
    double volume_per_node = (2*8*l_num_req*cores_per_node)*(1.0E-9);
    double injection_bw = (cores_per_node > 0) ? (volume_per_node / laptime) : 0.0;

    T0_fprintf(stderr,"  %8.3lf seconds\n", laptime);
    (void)buf_cnt; (void)models_mask; (void)printhelp; (void)injection_bw;

    num_errors += ig_check_and_zero(0, tgt, index, l_num_req);
    total_errors = num_errors;
    if (total_errors) {
      T0_fprintf(stderr,"YOU FAILED!!!!\n");
    }

    lgp_barrier();
    lgp_all_free(table);
    free(index);
    free(pckindx);
    free(tgt);

  });
  return 0;
}
