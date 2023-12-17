package main

import (
	"fmt"
	"time"

	"github.com/anacrolix/torrent"
	"github.com/dustin/go-humanize"
	"github.com/rivo/tview"
)

type TorrentProgress struct {
	Name 		string
	Size 		string
	Status 		string
	DownSpeed	string
	UpSpeed		string
	Seeds		string
	Peers 		string
	Completed 	string
}

type TorrentManager struct {
	tview.TableContentReadOnly

	table *tview.Table
	torrents []TorrentProgress
}

func (d *TorrentManager) GetCell(row, column int) *tview.TableCell {

	t := d.torrents[row]
	switch column {
	case 0:
		return tview.NewTableCell(t.Name)
	case 1:
		return tview.NewTableCell(t.Status)
	case 2:
		return tview.NewTableCell(t.Size)
	case 3:
		return tview.NewTableCell(t.DownSpeed)
	case 4:
		return tview.NewTableCell(t.UpSpeed)
	case 5:
		return tview.NewTableCell(t.Seeds)
	case 6:
		return tview.NewTableCell(t.Peers)
	case 7:
		return tview.NewTableCell(t.Completed)
	}
	return nil
}

func (d *TorrentManager) GetRowCount() int {
	return len(d.torrents)
}

func (d *TorrentManager) GetColumnCount() int {
	return 8
}

func (d *TorrentManager) Download(t *torrent.Torrent, update func()) {

	d.torrents = append(d.torrents, TorrentProgress{
		Name: t.Name(),
		Size: humanize.Bytes(uint64(t.Length())),
	})

	go func(index int) {
		tp := &d.torrents[index]
		if t.Info() == nil {
			tp.Status = "Getting torrent info"
			<-t.GotInfo()
			t.DownloadAll()
			tp.Name = t.Name()
			tp.Status = "Downloading"
		}
		lastStats := t.Stats()
		interval := 1 * time.Second
		for range time.Tick(interval) {
			stats := t.Stats()
			byteRate := int64(time.Second)
			byteRate *= stats.BytesReadUsefulData.Int64() - lastStats.BytesReadUsefulData.Int64()
			byteRate /= int64(interval)
			tp.DownSpeed = humanize.Bytes(uint64(byteRate)) + "/s"

			byteRate = int64(time.Second)
			byteRate *= stats.BytesWritten.Int64() - lastStats.BytesWritten.Int64()
			byteRate /= int64(interval)
			tp.UpSpeed = humanize.Bytes(uint64(byteRate)) + "/s"

			tp.Size = humanize.Bytes(uint64(t.Length()))
			tp.Seeds = fmt.Sprintf("%d", stats.ConnectedSeeders)
			tp.Peers = fmt.Sprintf("%d", stats.TotalPeers)
			tp.Completed = humanize.Bytes(uint64(t.BytesCompleted()))

			lastStats = stats

			update()
		}
	}(len(d.torrents)-1)
}

func main() {

	clientConfig := torrent.NewDefaultClientConfig()
	client, _ := torrent.NewClient(clientConfig)
	app := tview.NewApplication()

	data := &TorrentManager{
		torrents: []TorrentProgress{
			TorrentProgress{
				Name: "Name",
				Status: "Status",
				Size: "Size",
				DownSpeed: "Down Speed",
				UpSpeed: "Up Speed",
				Seeds: "Seeds",
				Peers: "Peers",
				Completed: "Completed",
			},
		},
	}

	t, _ := client.AddMagnet("magnet:?xt=urn:btih:28a399dc14f6ff3d37e975b072da4095fe7357e9&dn=archlinux-2023.12.01-x86_64.iso")
	data.Download(t, func() {
		app.Draw()
	})

	t, _ = client.AddMagnet("magnet:?xt=urn:btih:KRWPCX3SJUM4IMM4YF5RPHL6ANPYTQPU")
	data.Download(t, func() {
		app.Draw()
	})


	table := tview.NewTable().
		SetBorders(true).
		SetSelectable(true, true).
		SetContent(data)

	data.table = table

	if err := app.SetRoot(table, true).EnableMouse(false).Run(); err != nil {
		panic(err)
	}
}