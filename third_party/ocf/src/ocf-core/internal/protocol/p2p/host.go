package p2p

import (
	"context"
	"crypto/rand"
	"fmt"
	mrand "math/rand"
	"ocfcore/internal/common"
	"strconv"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	p2phttp "github.com/libp2p/go-libp2p-http"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/protocol"
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"
	"github.com/libp2p/go-libp2p/p2p/net/connmgr"
	"github.com/libp2p/go-libp2p/p2p/security/noise"
	libp2ptls "github.com/libp2p/go-libp2p/p2p/security/tls"
	"github.com/spf13/viper"
)

var P2PNode *host.Host
var once sync.Once

func GetP2PNode() host.Host {
	once.Do(func() {
		ctx := context.Background()
		var err error
		seed := viper.GetString("seed")
		// try to parse the seed as int64
		seedInt, err := strconv.ParseInt(seed, 10, 64)
		if err != nil {
			panic(err)
		}
		host, err := newHost(ctx, seedInt)
		P2PNode = &host
		if err != nil {
			panic(err)
		}
	})
	return *P2PNode
}

// Only the native HTTP protocol receives an explicit, bounded override.
// A zero stream setting preserves libp2p's original autoscaled defaults.
func httpResourceOption(streams, memoryMiB int) (libp2p.Option, error) {
	if streams == 0 {
		return nil, nil
	}
	if streams < 1 || streams > 4096 || memoryMiB < 16 || memoryMiB > 4096 {
		return nil, fmt.Errorf("invalid p2p HTTP resource limits: streams must be 1..4096 and memory MiB 16..4096")
	}
	defaults := rcmgr.DefaultLimits
	libp2p.SetDefaultServiceLimits(&defaults)
	limits := rcmgr.PartialLimitConfig{
		ProtocolPeer: map[protocol.ID]rcmgr.ResourceLimits{
			p2phttp.DefaultP2PProtocol: {
				Streams:         rcmgr.LimitVal(streams),
				StreamsInbound:  rcmgr.LimitVal(streams),
				StreamsOutbound: rcmgr.LimitVal(streams),
				Memory:          rcmgr.LimitVal64(int64(memoryMiB) << 20),
			},
		},
	}.Build(defaults.AutoScale())
	manager, err := rcmgr.NewResourceManager(rcmgr.NewFixedLimiter(limits))
	if err != nil {
		return nil, err
	}
	return libp2p.ResourceManager(manager), nil
}

func newHost(ctx context.Context, seed int64) (host.Host, error) {
	streams := viper.GetInt("p2p.http_streams_per_peer")
	memoryMiB := viper.GetInt("p2p.http_memory_mib_per_peer")
	resourceOption, err := httpResourceOption(streams, memoryMiB)
	if err != nil {
		return nil, err
	}
	connmgr, err := connmgr.NewConnManager(
		100, // Lowwater
		400, // HighWater,
		connmgr.WithGracePeriod(time.Minute),
	)
	if err != nil {
		common.Logger.Error("Error while creating connection manager: %v", err)
	}
	var priv crypto.PrivKey
	fmt.Println("seed: ", seed)
	// try to load the private key from file
	if seed == 0 {
		// try to load from the file
		priv = loadKeyFromFile()
		if priv == nil {
			r := rand.Reader
			priv, _, err = crypto.GenerateKeyPairWithReader(crypto.RSA, 2048, r)
			if err != nil {
				return nil, err
			}
		}
	} else {
		r := mrand.New(mrand.NewSource(seed))
		priv, _, err = crypto.GenerateKeyPairWithReader(crypto.RSA, 2048, r)
		if err != nil {
			return nil, err
		}
	}
	// persist private key
	writeKeyToFile(priv)
	if err != nil {
		return nil, err
	}

	options := []libp2p.Option{
		libp2p.DefaultTransports,
		libp2p.Identity(priv),
		libp2p.ConnectionManager(connmgr),
		libp2p.NATPortMap(),
		libp2p.ListenAddrStrings(
			"/ip4/0.0.0.0/tcp/43905",
			"/ip4/0.0.0.0/udp/59820/quic",
		),
		libp2p.Security(libp2ptls.ID, libp2ptls.New),
		libp2p.Security(noise.ID, noise.New),
		libp2p.EnableNATService(),
		libp2p.EnableRelay(),
		libp2p.EnableHolePunching(),
		libp2p.ForceReachabilityPublic(),
	}
	if resourceOption != nil {
		options = append(options, resourceOption)
		fmt.Printf("OCF HTTP protocol per-peer limits: inbound=%d outbound=%d total=%d memory_mib=%d\n", streams, streams, streams, memoryMiB)
	}
	return libp2p.New(options...)
}
