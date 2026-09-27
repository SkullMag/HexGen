package server

import (
	"encoding/json"
	"net/http"
)

func ErrorHandler(res http.ResponseWriter, req *http.Request, err error) {
	res.Header().Set("Content-Type", "application/json")
	res.WriteHeader(http.StatusBadGateway)
	json.NewEncoder(res).Encode(map[string]string{"error": err.Error()})
}
