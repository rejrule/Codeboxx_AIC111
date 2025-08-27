import json, os, torch, torch.nn as nn

MODEL_PATH = os.environ.get("MODEL_PATH", "/var/task/model.pth")

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

model = TinyNet()
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    except Exception:
        pass
else:
    with torch.no_grad():
        model.linear.weight.fill_(3.0)   # y = 3x + 1
        model.linear.bias.fill_(1.0)
model.eval()

def _predict(x: float) -> float:
    with torch.no_grad():
        y = model(torch.tensor([[x]], dtype=torch.float32))
        return float(y.item())

def lambda_handler(event, context):
    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        x = float(json.loads(body)["x"])
        y = _predict(x)
        return {"statusCode": 200,
                "headers": {"Content-Type":"application/json"},
                "body": json.dumps({"prediction": y})}
    except Exception as e:
        return {"statusCode": 400,
                "headers": {"Content-Type":"application/json"},
                "body": json.dumps({"error": str(e)})}
