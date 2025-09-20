#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace std;

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using ms = std::chrono::milliseconds;

#ifndef ENABLE_VOL_SIGNAL
#define ENABLE_VOL_SIGNAL 1   // set to 0 to disable bonus volatility signal
#endif

// --------------------------- Market Data ---------------------------
struct alignas(64) MarketData {
    int instrument_id;
    double price;
    Clock::time_point timestamp;
};

class MarketDataFeed {
public:
    MarketDataFeed(std::vector<MarketData>& ref, int n_instruments = 10)
        : data(ref), num_instruments(n_instruments) {}

    void generateData(int num_ticks) {
        std::mt19937_64 gen(0xC0FFEE);
        // Price paths per instrument use a mild mean-reverting random walk
        std::vector<double> px(num_instruments, 150.0);
        std::normal_distribution<double> shock(0.0, 0.5); // ~50 bps noise
        std::normal_distribution<double> drift(0.0, 0.02);

        data.clear();
        data.reserve(num_ticks);
        for (int i = 0; i < num_ticks; ++i) {
            int id = i % num_instruments;
            // OU-style update toward 150 with small noise
            double kappa = 0.02;
            double mu = 150.0 + drift(gen);
            px[id] += kappa * (mu - px[id]) + shock(gen);
            MarketData md;
            md.instrument_id = id;
            md.price = std::clamp(px[id], 50.0, 500.0);
            md.timestamp = Clock::now();
            data.emplace_back(md);
        }
    }

private:
    std::vector<MarketData>& data;
    int num_instruments;
};

// --------------------------- Orders ---------------------------
struct alignas(64) Order {
    int instrument_id;
    double price;
    bool is_buy;
    uint32_t signal_mask; // bitmask of which signals fired
    Clock::time_point timestamp; // send time
};

// --------------------------- Utilities ---------------------------
template<size_t CAP>
struct PriceHistory {
    array<double, CAP> buf{};
    size_t size = 0;
    size_t head = 0; // next write
    double sum = 0.0;
    double sumsq = 0.0;

    inline void add(double x) {
        if (size < CAP) {
            buf[head] = x;
            head = (head + 1) % CAP;
            size++;
            sum += x;
            sumsq += x * x;
        } else {
            double old = buf[head];
            sum -= old; sumsq -= old * old;
            buf[head] = x;
            head = (head + 1) % CAP;
            sum += x; sumsq += x * x;
        }
    }
    inline bool full() const { return size == CAP; }

    inline double avg() const { return size ? (sum / double(size)) : 0.0; }

    inline double stddev() const {
        if (size < 2) return 0.0;
        double m = avg();
        double var = (sumsq / double(size)) - m * m;
        return var > 0 ? std::sqrt(var) : 0.0;
    }

    inline bool last(double& out) const {
        if (!size) return false;
        size_t idx = (head + CAP - 1) % CAP;
        out = buf[idx];
        return true;
    }

    inline bool last2(double& prev, double& lastv) const {
        if (size < 2) return false;
        size_t i1 = (head + CAP - 2) % CAP;
        size_t i2 = (head + CAP - 1) % CAP;
        prev = buf[i1]; lastv = buf[i2];
        return true;
    }

    inline bool last3(double& a, double& b, double& c) const {
        if (size < 3) return false;
        size_t i0 = (head + CAP - 3) % CAP;
        size_t i1 = (head + CAP - 2) % CAP;
        size_t i2 = (head + CAP - 1) % CAP;
        a = buf[i0]; b = buf[i1]; c = buf[i2];
        return true;
    }
};

// --------------------------- Trading Engine ---------------------------
class TradeEngine {
public:
    explicit TradeEngine(const std::vector<MarketData>& feed, int n_instruments = 10)
        : market_data(feed),
          price_hist(n_instruments),
          per_signal_counts{0,0,0,0} {
        orders.reserve(feed.size() / 10); // heuristic
        latencies.reserve(feed.size() / 5);
    }

    void process() {
        for (const auto& tick : market_data) {
            auto& hist = price_hist[tick.instrument_id];
            hist.add(tick.price);

            // signals -> accumulate buy/sell "votes"
            int buy_votes = 0, sell_votes = 0;
            uint32_t mask = 0;

            // Signal 1: Absolute thresholds (buy low, sell high)
            if (signal1(tick, buy_votes, sell_votes)) mask |= (1u << 0);

            // Signal 2: Deviation from rolling average (mean reversion)
            if (signal2(tick, buy_votes, sell_votes, hist)) mask |= (1u << 1);

            // Signal 3: Simple momentum (2 consecutive moves)
            if (signal3(tick, buy_votes, sell_votes, hist)) mask |= (1u << 2);

#if ENABLE_VOL_SIGNAL
            // Bonus Signal 4: Volatility breakout
            if (signal4_vol_breakout(tick, buy_votes, sell_votes, hist)) mask |= (1u << 3);
#endif

            if (buy_votes || sell_votes) {
                bool is_buy = (buy_votes > sell_votes) || (buy_votes == sell_votes && (tick.instrument_id & 1));
                auto now = Clock::now();
                double px = tick.price + (is_buy ? 0.01 : -0.01);
                orders.push_back(Order{tick.instrument_id, px, is_buy, mask, now});

                auto latency = std::chrono::duration_cast<ns>(now - tick.timestamp).count();
                latencies.push_back(latency);

                // track per-signal contributions (if that bit fired, attribute this order too)
                for (int s = 0; s < 4; ++s)
                    if (mask & (1u << s)) per_signal_counts[s]++;
            }
        }
    }

    void reportStats() const {
        long long sum = 0, max_latency = 0;
        for (auto l : latencies) { sum += l; if (l > max_latency) max_latency = l; }

        // p50 / p95 / p99 (optional but handy)
        vector<long long> sorted = latencies;
        auto pct = [&](double p)->long long {
            if (sorted.empty()) return 0;
            size_t idx = size_t(p * (sorted.size()-1));
            nth_element(sorted.begin(), sorted.begin()+idx, sorted.end());
            return sorted[idx];
        };

        cout << "\n--- Performance Report ---\n";
        cout << "Total Market Ticks Processed: " << market_data.size() << "\n";
        cout << "Total Orders Placed: " << orders.size() << "\n";
        cout << "Average Tick-to-Trade Latency (ns): " << (latencies.empty() ? 0 : sum / (long long)latencies.size()) << "\n";
        cout << "Max Tick-to-Trade Latency (ns): " << max_latency << "\n";
        if (!latencies.empty()) {
            cout << "p50/p95/p99 Latency (ns): " << pct(0.50) << " / " << pct(0.95) << " / " << pct(0.99) << "\n";
        }
        cout << "\nPer-signal order attributions (orders where the signal fired):\n";
        cout << "  S1 Threshold     : " << per_signal_counts[0] << "\n";
        cout << "  S2 MeanRevert    : " << per_signal_counts[1] << "\n";
        cout << "  S3 Momentum      : " << per_signal_counts[2] << "\n";
#if ENABLE_VOL_SIGNAL
        cout << "  S4 VolBreakout   : " << per_signal_counts[3] << "\n";
#endif
    }

    // Bonus: write CSV of orders
    void exportCSV(const std::string& path = "orders.csv") const {
        std::ofstream f(path);
        if (!f) return;
        f << "instrument_id,price,is_buy,signal_mask,send_time_ns\n";
        for (const auto& o : orders) {
            auto t = std::chrono::time_point_cast<ns>(o.timestamp).time_since_epoch().count();
            f << o.instrument_id << "," << std::fixed << std::setprecision(5) << o.price << ","
              << (o.is_buy ? 1 : 0) << "," << o.signal_mask << "," << t << "\n";
        }
    }

    // expose counts for the write-up
    const array<size_t,4>& signalCounts() const { return per_signal_counts; }

private:
    const std::vector<MarketData>& market_data;
    std::vector<Order> orders;
    std::vector<long long> latencies;
    std::vector<PriceHistory<32>> price_hist; // small, cache-friendly window
    array<size_t,4> per_signal_counts;

    // --------- Signals ----------
    // S1: Absolute thresholds (buy low, sell high)
    inline bool signal1(const MarketData& tick, int& buy, int& sell) const {
        if (tick.price < 105.0) { buy++; return true; }
        if (tick.price > 195.0) { sell++; return true; }
        return false;
    }

    // S2: Deviation from rolling average (mean reversion)
    inline bool signal2(const MarketData& tick, int& buy, int& sell, const PriceHistory<32>& hist) const {
        if (hist.size < 5) return false;
        double avg = hist.avg();
        if (avg == 0.0) return false;
        if (tick.price < avg * 0.98) { buy++; return true; }
        if (tick.price > avg * 1.02) { sell++; return true; }
        return false;
    }

    // S3: Momentum (two consecutive moves same direction)
    inline bool signal3(const MarketData& tick, int& buy, int& sell, const PriceHistory<32>& hist) const {
        double a,b,c;
        if (!hist.last3(a,b,c)) return false;
        double d1 = b - a, d2 = c - b;
        if (d1 > 0 && d2 > 0) { buy++; return true; }
        if (d1 < 0 && d2 < 0) { sell++; return true; }
        return false;
    }

#if ENABLE_VOL_SIGNAL
    // S4 (bonus): Volatility breakout vs recent stdev
    inline bool signal4_vol_breakout(const MarketData& tick, int& buy, int& sell, const PriceHistory<32>& hist) const {
        if (hist.size < 12) return false;
        double sd = hist.stddev();
        double prev;
        if (!hist.last(prev)) return false;
        double chg = tick.price - prev;
        double k = 1.75; // breakout multiplier
        if (sd <= 1e-9) return false;
        if (chg >  k * sd) { buy++; return true; }
        if (chg < -k * sd) { sell++; return true; }
        return false;
    }
#endif
};

// --------------------------- Main ---------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<MarketData> feed;
    MarketDataFeed generator(feed);

    auto start = Clock::now();
    generator.generateData(100000);

    TradeEngine engine(feed);
    engine.process();

    auto end = Clock::now();
    auto runtime = std::chrono::duration_cast<ms>(end - start).count();

    engine.reportStats();
    engine.exportCSV("orders.csv"); // bonus

    cout << "Total Runtime (ms): " << runtime << "\n";
    return 0;
}
