#pragma once
#include <arpa/inet.h>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace srs_iq {

inline thread_local uint16_t cur_rnti = 0; // set by uplink_processor before estimate()

constexpr unsigned MAX_SC = 1632; // 272 PRB, comb 2

#pragma pack(push, 1)
struct header {
  char     magic[4];   // "SRSQ"
  uint16_t version;    // 1
  uint16_t rnti;
  uint16_t sfn;
  uint8_t  slot;
  uint8_t  numerology;
  uint8_t  rx_port;
  uint8_t  start_symbol;
  uint8_t  nof_symbols;
  uint8_t  comb;
  uint16_t first_sc;   // initial subcarrier of the comb
  uint16_t n_sc;       // number of complex samples that follow
  float    ta_us;      // estimated time alignment (already compensated in the samples)
  float    epre_db;
  float    rsrp_db;
  float    noise_var;
  double   host_time;  // CLOCK_REALTIME, seconds
};
#pragma pack(pop)

class sender {
  int         fd = -1;
  sockaddr_in dst{};

public:
  sender()
  {
    const char* p = std::getenv("SRS_IQ_PORT");
    if (!p) return;
    fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(static_cast<uint16_t>(std::atoi(p)));
    inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);
  }
  ~sender() { if (fd >= 0) ::close(fd); }
  bool enabled() const { return fd >= 0; }

  void send(header h, const std::complex<float>* iq, unsigned n)
  {
    if (fd < 0) return;
    if (n > MAX_SC) n = MAX_SC;
    alignas(8) uint8_t buf[sizeof(header) + MAX_SC * sizeof(std::complex<float>)];
    std::memcpy(h.magic, "SRSQ", 4);
    h.version = 1;
    h.n_sc    = static_cast<uint16_t>(n);
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    h.host_time = ts.tv_sec + ts.tv_nsec * 1e-9;
    std::memcpy(buf, &h, sizeof(h));
    std::memcpy(buf + sizeof(h), iq, n * sizeof(std::complex<float>));
    ::sendto(fd, buf, sizeof(h) + n * sizeof(std::complex<float>), MSG_DONTWAIT,
             reinterpret_cast<const sockaddr*>(&dst), sizeof(dst));
  }
};

inline sender& get()
{
  static thread_local sender s; // one socket per PHY worker thread, no locking
  return s;
}

} // namespace srs_iq
