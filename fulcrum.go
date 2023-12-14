package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"sync"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/core/host"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
	dutil "github.com/libp2p/go-libp2p/p2p/discovery/util"

	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/multiformats/go-multiaddr"

	"github.com/ipfs/go-log/v2"


	"github.com/jmorganca/ollama/api"
)

var logger = log.Logger("rendezvous")

type generateOptions struct {
	Model    string
	Prompt   string
	Format   string
	Options  map[string]interface{}
}

func ConnectOllama( host host.Host, opts generateOptions ) {

	// create a client to the locally running ollama server
	ollama, err := api.ClientFromEnvironment()
	if err != nil {
		panic(err)
	}

	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// pull the model even if it exists
	pr := api.PullRequest{Name: opts.Model}
	if err := ollama.Pull(context.Background(), &pr, func(resp api.ProgressResponse) error {
		fmt.Println("pulling...", resp.Completed, resp.Total)
		return nil
	}); err != nil {
		panic(err)
	}

	// Set a function as stream handler. This function is called when a peer
	// initiates a connection and starts a stream with this peer.
	host.SetStreamHandler(protocol.ID("ollama/" + opts.Model), func (stream network.Stream) {
		logger.Info("!!! Got a new stream !!!")
	
		// get the prompt request from peer
		var request api.GenerateRequest
		if err := json.NewDecoder(stream).Decode(&request); err != nil {
			logger.Error(err)
			return
		}
	
		logger.Info("Received request: ", request.Prompt)

		// generate an answer
		if err := ollama.Generate(cancelCtx, &request, func(response api.GenerateResponse) error {

			if err := json.NewEncoder(stream).Encode(response); err != nil {
				logger.Error(err)
				panic(err)
			}
	
			return nil

		}); err != nil {
			panic(err)
		}
	
	})
}


func main() {
	//log.SetAllLoggers(log.LevelWarn)
	log.SetLogLevel("rendezvous", "info")
	help := flag.Bool("h", false, "Display Help")
	config, err := ParseFlags()
	if err != nil {
		panic(err)
	}

	if *help {
		fmt.Println("This program demonstrates a simple p2p chat application using libp2p")
		fmt.Println()
		fmt.Println("Usage: Run './chat in two different terminals. Let them connect to the bootstrap nodes, announce themselves and connect to the peers")
		flag.PrintDefaults()
		return
	}

	// libp2p.New constructs a new libp2p Host. Other options can be added
	// here.
	host, err := libp2p.New(libp2p.ListenAddrs([]multiaddr.Multiaddr(config.ListenAddresses)...))
	if err != nil {
		panic(err)
	}
	logger.Info("Host created. We are:", host.ID())
	logger.Info(host.Addrs())

	ConnectOllama(host, generateOptions{
		Model: "orca2:13b",
		Prompt: "What is it?",
		Options: map[string]interface{}{},
	})

	// Start a DHT, for use in peer discovery. We can't just make a new DHT
	// client because we want each peer to maintain its own local copy of the
	// DHT, so that the bootstrapping node of the DHT can go down without
	// inhibiting future peer discovery.
	ctx := context.Background()
	kademliaDHT, err := dht.New(ctx, host)
	if err != nil {
		panic(err)
	}

	// Bootstrap the DHT. In the default configuration, this spawns a Background
	// thread that will refresh the peer table every five minutes.
	logger.Debug("Bootstrapping the DHT")
	if err = kademliaDHT.Bootstrap(ctx); err != nil {
		panic(err)
	}

	// Let's connect to the bootstrap nodes first. They will tell us about the
	// other nodes in the network.
	var wg sync.WaitGroup
	for _, peerAddr := range config.BootstrapPeers {
		peerinfo, _ := peer.AddrInfoFromP2pAddr(peerAddr)
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := host.Connect(ctx, *peerinfo); err != nil {
				logger.Warning(err)
			} else {
				logger.Info("Connection established with bootstrap node:", *peerinfo)
			}
		}()
	}
	wg.Wait()

	// We use a rendezvous point "meet me here" to announce our location.
	// This is like telling your friends to meet you at the Eiffel Tower.
	logger.Info("Announcing ourselves...", config.RendezvousString)
	routingDiscovery := drouting.NewRoutingDiscovery(kademliaDHT)
	dutil.Advertise(ctx, routingDiscovery, config.RendezvousString)
	logger.Debug("Successfully announced!")

	// Now, look for others who have announced
	// This is like your friend telling you the location to meet you.
	logger.Debug("Searching for other peers...")
	peerChan, err := routingDiscovery.FindPeers(ctx, config.RendezvousString)
	if err != nil {
		panic(err)
	}

	for peer := range peerChan {
		if peer.ID == host.ID() {
			continue
		}
		logger.Debug("Found peer:", peer)

		logger.Debug("Connecting to:", peer)
		stream, err := host.NewStream(ctx, peer.ID, protocol.ID(config.ProtocolID))

		if err != nil {
			//logger.Warning("Connection failed:", err)
			continue
		} else {

			var request api.GenerateRequest
			request.Prompt = "What is it?"
			if err := json.NewEncoder(stream).Encode(request); err != nil {
				logger.Error(err)
				break
			}

			var response api.GenerateResponse
			if err := json.NewDecoder(stream).Decode(&response); err != nil {
				logger.Error(err)
				break
			}

			logger.Info("Received response: ", response.Response)

		}

		logger.Info("Connected to:", peer)
	}

	select {}
}
