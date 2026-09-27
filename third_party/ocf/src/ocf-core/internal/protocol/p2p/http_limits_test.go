package p2p

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	gostream "github.com/libp2p/go-libp2p-gostream"
	p2phttp "github.com/libp2p/go-libp2p-http"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"
)

// Isolate the protocol-per-peer boundary from the RAM-scaled parent peer limit.
// Production keeps all parent limits unchanged; a smaller host may hit them first.
type testParentStreamLimit struct{ rcmgr.Limit }

func (testParentStreamLimit) GetStreamLimit(network.Direction) int { return 2048 }
func (testParentStreamLimit) GetStreamTotalLimit() int             { return 2048 }

func TestHTTPResourceOverrideBounds(t *testing.T) {
	opt, err := httpResourceOption(0, 0)
	if err != nil || opt != nil {
		t.Fatal("zero setting must preserve original defaults")
	}
	for _, pair := range [][2]int{{-1, 512}, {4097, 512}, {1024, 0}, {1024, 4097}} {
		if _, err := httpResourceOption(pair[0], pair[1]); err == nil {
			t.Fatalf("accepted invalid limits: %v", pair)
		}
	}
	opt, err = httpResourceOption(1024, 512)
	if err != nil {
		t.Fatal(err)
	}
	h, err := libp2p.New(opt, libp2p.NoListenAddrs)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	err = h.Network().ResourceManager().ViewPeer(peer.ID("bounded-test-peer"), func(scope network.PeerScope) error {
		limiter := scope.(rcmgr.ResourceScopeLimiter)
		limiter.SetLimit(testParentStreamLimit{limiter.Limit()})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var held []network.StreamManagementScope
	defer func() {
		for _, s := range held {
			s.Done()
		}
	}()
	for i := 0; i < 1024; i++ {
		s, err := h.Network().ResourceManager().OpenStream(peer.ID("bounded-test-peer"), network.DirInbound)
		if err != nil {
			t.Fatalf("stream %d: %v", i, err)
		}
		if err = s.SetProtocol(p2phttp.DefaultP2PProtocol); err != nil {
			s.Done()
			t.Fatalf("protocol stream %d: %v", i, err)
		}
		held = append(held, s)
	}
	s, err := h.Network().ResourceManager().OpenStream(peer.ID("bounded-test-peer"), network.DirInbound)
	if err == nil {
		defer s.Done()
		if err = s.SetProtocol(p2phttp.DefaultP2PProtocol); err == nil {
			t.Fatal("1025th inbound stream must be rejected")
		}
	}
}

// Exercise the actual native HTTP-over-libp2p transport, with requests held open
// until all 500 have reached the server. No inference workers are involved.
func TestHTTPTransport500Concurrent(t *testing.T) {
	serverOpt, err := httpResourceOption(1024, 512)
	if err != nil {
		t.Fatal(err)
	}
	clientOpt, err := httpResourceOption(1024, 512)
	if err != nil {
		t.Fatal(err)
	}
	serverHost, err := libp2p.New(serverOpt, libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatal(err)
	}
	defer serverHost.Close()
	clientHost, err := libp2p.New(clientOpt, libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatal(err)
	}
	defer clientHost.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if err = clientHost.Connect(ctx, peer.AddrInfo{ID: serverHost.ID(), Addrs: serverHost.Addrs()}); err != nil {
		t.Fatal(err)
	}
	listener, err := gostream.Listen(serverHost, p2phttp.DefaultP2PProtocol)
	if err != nil {
		t.Fatal(err)
	}
	arrived := make(chan struct{}, 500)
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		arrived <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
		}
		io.WriteString(w, "ok")
	})}
	defer server.Close()
	go server.Serve(listener)
	client := &http.Client{Transport: p2phttp.NewTransport(clientHost)}
	done := make(chan error, 500)
	for i := 0; i < 500; i++ {
		go func() {
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "libp2p://"+serverHost.ID().String()+"/hold", nil)
			resp, err := client.Do(req)
			if err == nil {
				_, err = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode != 200 {
					err = fmt.Errorf("HTTP %d", resp.StatusCode)
				}
			}
			done <- err
		}()
	}
	for i := 0; i < 500; i++ {
		select {
		case <-arrived:
		case err := <-done:
			t.Fatalf("request failed before all 500 were admitted: %v", err)
		case <-ctx.Done():
			t.Fatalf("only %d/500 admitted: %v", i, ctx.Err())
		}
	}
	releaseOnce.Do(func() { close(release) })
	for i := 0; i < 500; i++ {
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	t.Log("All 500 concurrent native HTTP streams admitted and completed successfully")
}
